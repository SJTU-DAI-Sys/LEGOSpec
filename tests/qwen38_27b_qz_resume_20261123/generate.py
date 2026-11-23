"""SGLang chat generation with strict per-turn thinking caps and ID resume."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from itertools import chain
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import zlib

JOB = Path(__file__).resolve().parent

class Rejected(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason, self.details = reason, details

def http(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def template_kwargs(config):
    return dict(enable_thinking=True, preserve_thinking=config['preserve_thinking'],
                reasoning_effort=config['reasoning_effort'])

def validate_template(config, rendered):
    assert rendered.endswith('<|im_start|>assistant\n<think>\n'), 'Thinking prefix missing'
    effort=config['reasoning_effort']
    assert effort in ('low','medium','xhigh'), effort
    for marker_effort in ('low','xhigh'):
        marker=f'Reasoning effort is set to {marker_effort}.'
        assert (marker in rendered)==(effort==marker_effort), (effort,marker)

def prompt_ids(tokenizer, messages, config):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        **template_kwargs(config), return_dict=False)
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if not isinstance(ids, list) or (ids and isinstance(ids[0], list)):
        raise ValueError('Tokenizer must return flat input IDs')
    return ids

def seed_for(config, record_id, turn, attempt):
    value = f'{config["seed"]}|{record_id}|{turn}|{attempt}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'little') % (2**31-1)

def fingerprint(config):
    import importlib.metadata
    import importlib.util
    model = Path(config['model'])
    hashes = {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
              for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'model.safetensors.index.json']}
    hashes['generation_client.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec=importlib.util.find_spec('sglang')
    if spec and spec.origin:
        source=Path(spec.origin).parent
        for name in ['srt/models/qwen3_5.py','srt/configs/model_config.py','srt/server_args.py',
            'srt/entrypoints/openai/protocol.py','srt/entrypoints/openai/serving_chat.py',
            'srt/constrained/grammar_manager.py','srt/constrained/reasoner_grammar_backend.py']:
            path=source/name
            if path.exists():hashes['sglang/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['transformers_version']=importlib.metadata.version('transformers')
    return hashlib.sha256(json.dumps(dict(config=config, files=hashes), sort_keys=True).encode()).hexdigest(), hashes

def quality(text):
    if not isinstance(text, str) or not text.strip():
        raise Rejected('empty_response')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 20:
        common, count = Counter(lines).most_common(1)[0]
        # Ordinary numbered lists and repeated code punctuation are not a
        # failure by themselves. Require a dominant line and many repeats.
        if len(common) >= 4 and count >= 15 and count / len(lines) >= .6:
            raise Rejected('repeated_line_loop', repeated_line=common[:120], repeats=count)
    encoded = text.encode()
    if len(text) >= 1500 and len(zlib.compress(encoded))/len(encoded) < .12:
        raise Rejected('highly_repetitive_output')
    # Catch phrase loops even when no newline is present.
    if len(text) >= 1000 and re.search(r'(.{20,200}?)\1{7,}', text, re.DOTALL):
        raise Rejected('repeated_phrase_loop')

def request_payload(config, messages, record_id, turn, attempt=0, thinking_budget=None):
    params = {name:config[name] for name in ['temperature','top_p','top_k','min_p',
                                           'presence_penalty','repetition_penalty']}
    return dict(model='qwen38-27b-regen', messages=messages, stream=False, n=1,
        max_tokens=config['max_new_tokens'], seed=seed_for(config,record_id,turn,attempt),
        reasoning_effort=config['reasoning_effort'], chat_template_kwargs=template_kwargs(config),
        custom_params={'thinking_budget':config['thinking_budget'] if thinking_budget is None else thinking_budget},
        stop_token_ids=config['stop_token_ids'], ignore_eos=False,
        return_output_ids_in_sglext=True, **params)

def parse_response(config, tokenizer, endpoint, response, payload, local_prompt_tokens):
    try:
        choice=response['choices'][0]
        message=choice['message']
        output_ids=response['sglext']['output_ids'][0]
        usage=response['usage']
        assert isinstance(output_ids,list) and all(type(x) is int for x in output_ids)
    except (KeyError,TypeError,IndexError,AssertionError) as error:
        raise Rejected('invalid_server_response', error=str(error)) from error
    # Count the actual sampled IDs, not a re-tokenization of decoded text.
    end_ids=tokenizer.encode('</think>',add_special_tokens=False)
    close=next((i for i in range(len(output_ids)-len(end_ids)+1)
                if output_ids[i:i+len(end_ids)]==end_ids),None)
    detail=dict(prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
        sampled_tokens=len(output_ids), thinking_tokens_before_close=close,
        thinking_budget=payload['custom_params']['thinking_budget'],
        reasoning_effort=payload['reasoning_effort'], max_new_tokens=payload['max_tokens'],
        finish_reason=choice.get('finish_reason'), seed=payload['seed'], endpoint=endpoint,
        usage=usage, output_ids_sha256=hashlib.sha256(json.dumps(output_ids).encode()).hexdigest())
    if usage.get('prompt_tokens')!=local_prompt_tokens:
        raise Rejected('server_prompt_token_mismatch',local_tokens=local_prompt_tokens,generation=detail)
    if len(output_ids)>payload['max_tokens'] or usage.get('completion_tokens',0)>payload['max_tokens']:
        raise Rejected('output_budget_violated', generation=detail)
    if close is None:
        raise Rejected('missing_thinking_close',generation=detail,response=message)
    if close>payload['custom_params']['thinking_budget']:
        raise Rejected('thinking_budget_violated',generation=detail,response=message)
    reasoning=message.get('reasoning_content') or ''
    final=message.get('content')
    if choice.get('finish_reason')=='length':
        raise Rejected('truncated',generation=detail,response=message)
    if choice.get('finish_reason')!='stop':
        raise Rejected('unfinished_response',generation=detail,response=message)
    try:
        quality(final)
    except Rejected as error:
        error.details.update(generation=detail,response=message)
        raise
    # Chat history uses separate reasoning and final fields so the model's
    # preserve_thinking template inserts the delimiters exactly once.
    return dict(role='assistant',content=final,reasoning_content=reasoning),detail

def completion(config, tokenizer, endpoint, messages, record_id, turn, attempt=0, thinking_budget=None):
    ids = prompt_ids(tokenizer, messages, config)
    budget = config['max_new_tokens']
    # Preserve complete questions. Long requests are recorded separately;
    # they are never silently cropped to make them fit.
    if len(ids) + budget > config['context_length']:
        raise Rejected('context_length', prompt_tokens=len(ids), requested_new_tokens=budget,
                       context_length=config['context_length'])
    payload=request_payload(config,messages,record_id,turn,attempt,thinking_budget)
    response=http(endpoint+'/v1/chat/completions',payload,config['request_timeout'])
    return parse_response(config,tokenizer,endpoint,response,payload,len(ids))

def generate_record(config, tokenizer, endpoint, row):
    messages, saved_messages, generations = [], [], []
    users = 0
    for index, message in enumerate(row['conversations']):
        role = message.get('role')
        if role == 'assistant':
            continue
        if role not in ('system', 'user') or not isinstance(message.get('content'), str):
            raise Rejected('invalid_input_message', index=index)
        if not message['content'].strip():
            if role == 'user':
                continue
            raise Rejected('empty_system_message', index=index)
        messages.append(dict(role=role, content=message['content']))
        saved_messages.append(dict(role=role,content=message['content']))
        if role == 'system':
            continue
        users += 1
        last_error = None
        for attempt in range(config['retries'] + 1):
            try:
                assistant, meta = completion(config, tokenizer, endpoint, messages, row['id'], index, attempt)
                break
            except Rejected as error:
                last_error = error
                if error.reason in ('context_length','thinking_budget_violated','output_budget_violated',
                    'invalid_server_response','server_prompt_token_mismatch','missing_thinking_close'):
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = Rejected('transport_error', error=str(error))
                time.sleep(min(2**attempt, 10))
        else:
            raise last_error
        messages.append(assistant)
        full_text='<think>\n'+assistant['reasoning_content']+'\n</think>\n\n'+assistant['content']
        # Keep the native Qwen3.8 schema reusable with apply_chat_template:
        # the template, not content itself, wraps reasoning_content in <think>.
        saved_messages.append(dict(assistant,full_content=full_text,final_content=assistant['content']))
        generations.append(dict(original_user_index=index, attempt=attempt, **meta))
    if not users:
        raise Rejected('no_user_turn')
    return dict(row, conversations=saved_messages, status='success',
                generation=dict(protocol='sglang_chat_strict_thinking_v1', turns=generations,
                    sampling={k:config[k] for k in ['temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty',
                        'reasoning_effort','thinking_budget','max_new_tokens']},
                    enable_thinking=True,preserve_thinking=config['preserve_thinking']))

def verify_server(config, endpoint):
    info = http(endpoint + '/model_info')
    if str(info.get('model_path', '')).rstrip('/') != config['model'].rstrip('/'):
        raise ValueError(f'Unexpected model at {endpoint}: {info.get("model_path")}')
    return info

def smoke(config, tokenizer, endpoints, parts):
    proof = []
    for endpoint in endpoints:
        info = verify_server(config, endpoint)
        # A deliberately tiny per-request cap exercises the actual strict
        # decoding path, independently of the production global 2048 cap.
        for name,question,cap in [
            ('strict_budget8','Explain why there are infinitely many prime numbers. Include a proof.',8),
            ('math','What is 2 + 3? Reply with only the integer.',None),
            ('code','Write a Python function add(a, b) that returns a + b. No explanation.',None)]:
            assistant,meta=completion(config,tokenizer,endpoint,[dict(role='user',content=question)],
                'smoke/'+name,0,thinking_budget=cap)
            text=assistant['content']
            if name=='math' and text.strip().strip(' .\n')!='5':
                raise Rejected('smoke_math_wrong',answer=text)
            if name=='code' and not ('def ' in text and 'return' in text and '+' in text):
                raise Rejected('smoke_code_wrong',answer=text)
            proof.append(dict(endpoint=endpoint,case=name,assistant=assistant,generation=meta))
        messages=[dict(role='user',content='Remember the number 17. Reply with only OK.')]
        first,meta=completion(config,tokenizer,endpoint,messages,'smoke/multi',0)
        messages += [first,dict(role='user',content='What number did I ask you to remember? Reply with only the number.')]
        second,meta=completion(config,tokenizer,endpoint,messages,'smoke/multi',1)
        if second['content'].strip().strip(' .\n')!='17':
            raise Rejected('smoke_multi_wrong',answer=second['content'])
        proof.append(dict(endpoint=endpoint,case='multi',assistant=second,generation=meta))
    for part in parts:
        for index,row in enumerate(input_rows(config,part)):
            sample=generate_record(config,tokenizer,endpoints[(part+index)%len(endpoints)],row)
            proof.append(dict(case='real_input',part=part,id=row['id'],
                answer=next(m['content'] for m in sample['conversations'] if m['role']=='assistant')[:500],
                generation=sample['generation']))
            if index==1:break
    return proof

def input_rows(config,part):
    with Path(config['input_files'][part]).open() as stream:
        for line in stream:
            yield json.loads(line)

def open_output(path, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild success IDs from the append-only durable output. Interrupted
    # appends can be truncated; already fsynced successes are not regenerated.
    if path.exists():
        with path.open('rb+') as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    stream.truncate(position)
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    raise RuntimeError(f'Invalid committed output at byte {position}: {path}')
                if record['status'] != 'success':
                    raise ValueError('Non-success record in accepted output')
                for message in record['conversations']:
                    if message['role'] == 'assistant':
                        final=message['final_content']
                        if not isinstance(final,str) or not final.strip():
                            raise ValueError('Committed output has an empty final answer')
                done.add(json.dumps(record['id'],sort_keys=True))
    return path.open('a', buffering=1)

def run_part(config, tokenizer, endpoints, part, output_root, run_fingerprint):
    import fcntl
    import os
    lock=(output_root/f'part{part}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(config['input_files'][part])
    pin=output_root/f'part{part}.identity.json'
    stat=source.stat()
    identity=dict(fingerprint=run_fingerprint,input=str(source),input_bytes=stat.st_size,
        input_mtime_ns=stat.st_mtime_ns,part=part,input_shards=config['input_shards'])
    if pin.exists():
        assert json.loads(pin.read_text())==identity, 'Input/model/sampling changed; use a new RUN_TAG'
    else:
        assert not (output_root/f'part{part}.jsonl').exists(), 'Existing output without run identity; use a new RUN_TAG'
        pin.write_text(json.dumps(identity,indent=2)+'\n')
    done=set()
    output=open_output(output_root/f'part{part}.jsonl',done)
    rejected=(output_root/f'part{part}.rejected.jsonl').open('a',buffering=1)
    truncated=(output_root/f'part{part}.truncated.jsonl').open('a',buffering=1)
    # Continue unprocessed prompts before revisiting historical failures.
    # Failed rows remain queued for the second pass, with the same sampling.
    previous_failed=set()
    for failed_path in [output_root/f'part{part}.truncated.jsonl',output_root/f'part{part}.rejected.jsonl']:
        with failed_path.open() as stream:
            for line in stream:
                if line.endswith('\n'):
                    previous_failed.add(json.dumps(json.loads(line)['id'],sort_keys=True))
    print(json.dumps(dict(part=part,resumed_success=len(done),deferred_previous_failed=len(previous_failed))),flush=True)
    stats=Counter(resumed_success=len(done))
    # Per-worker queues bound memory even for large SWE prompts.
    limits={endpoint:threading.Semaphore(config['concurrency_per_replica']) for endpoint in endpoints}
    def task(row, endpoint):
        try:
            with limits[endpoint]:
                return 'success', generate_record(config,tokenizer,endpoint,row)
        except Rejected as error:
            if error.reason in ('thinking_budget_violated','output_budget_violated','invalid_server_response',
                'server_prompt_token_mismatch','missing_thinking_close'):
                raise RuntimeError(f'Backend contract failed: {error.reason}: {error.details}') from error
            return error.reason,dict(id=row['id'],status=error.reason,error=error.details)
        except Exception as error:
            # Preserve unexpected software failures for inspection; never
            # translate them into a successful sample.
            raise RuntimeError(f'Record {row.get("id")}: {error}') from error
    started=time.monotonic()
    def consume(future):
        kind, record=future.result()
        stats[kind]+=1
        handle=output if kind=='success' else truncated if kind=='truncated' else rejected
        handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        handle.flush()
        if kind=='success':done.add(json.dumps(record['id'],sort_keys=True))
        count=sum(v for k,v in stats.items() if k!='resumed_success')
        if count%config.get('fsync_every_records',32)==0:
            for stream in [output,rejected,truncated]:
                stream.flush();os.fsync(stream.fileno())
        if count%100==0:
            value=dict(part=part,counts=dict(stats),elapsed_seconds=round(time.monotonic()-started,1))
            temporary=output_root/f'part{part}.progress.tmp'
            temporary.write_text(json.dumps(value,indent=2)+'\n')
            temporary.replace(output_root/f'part{part}.progress.json')
            print(json.dumps(value),flush=True)
    with ThreadPoolExecutor(max_workers=len(endpoints)*config['concurrency_per_replica']) as pool:
        pending=set()
        sequence=0
        rows=chain(
            (row for row in input_rows(config,part) if json.dumps(row['id'],sort_keys=True) not in previous_failed),
            (row for row in input_rows(config,part) if json.dumps(row['id'],sort_keys=True) in previous_failed))
        for row in rows:
            if json.dumps(row['id'],sort_keys=True) in done:
                continue
            endpoint=endpoints[sequence%len(endpoints)]
            sequence+=1
            pending.add(pool.submit(task,row,endpoint))
            if len(pending)>=len(endpoints)*config['concurrency_per_replica']:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:consume(future)
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:consume(future)
    for stream in [output,rejected,truncated]:
        stream.flush();os.fsync(stream.fileno());stream.close()
    final=dict(part=part,status='completed',counts=dict(stats),accepted_total=len(done),elapsed_seconds=round(time.monotonic()-started,1))
    (output_root/f'part{part}.complete.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(JOB/'config.json'))
    parser.add_argument('--parts',default='0,1')
    parser.add_argument('--ports',default='32281,32282,32283,32284')
    parser.add_argument('--run-tag',default='default')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--reuse-smoke',action='store_true')
    args=parser.parse_args()
    assert not (args.smoke_only and args.reuse_smoke)
    config=json.loads(Path(args.config).read_text())
    parts=[int(s) for s in args.parts.split(',')]
    assert parts and len(parts)==len(set(parts)) and all(0<=p<config['input_shards'] for p in parts)
    endpoints=['http://127.0.0.1:'+str(int(p)) for p in args.ports.split(',')]
    assert re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_tag)
    output_root=Path(config['output_dir'])/args.run_tag
    output_root.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
    generation_config=json.loads((Path(config['model'])/'generation_config.json').read_text())
    assert set(generation_config['eos_token_id'])==set(config['stop_token_ids'])
    rendered=tokenizer.apply_chat_template([dict(role='user',content='test')],tokenize=False,
        add_generation_prompt=True,**template_kwargs(config))
    assert rendered.endswith('<|im_start|>assistant\n<think>\n'), 'Thinking chat template mismatch'
    validate_template(config,rendered)
    pin,hashes=fingerprint(config)
    receipt_path=output_root/('smoke.parts'+'-'.join(map(str,parts))+'.json')
    if args.reuse_smoke:
        receipt=json.loads(receipt_path.read_text())
        assert receipt['status']=='passed' and receipt['parts']==parts and receipt['endpoints']==endpoints
        assert receipt['fingerprint']==pin and receipt['model_metadata_sha256']==hashes, 'Smoke receipt does not match current model/client/config'
        for endpoint in endpoints:verify_server(config,endpoint)
        print('SMOKE REUSED: verified model/client/config receipt; all endpoint model paths rechecked',flush=True)
    else:
        smoke_proof=smoke(config,tokenizer,endpoints,parts)
        receipt=dict(status='passed',parts=parts,endpoints=endpoints,fingerprint=pin,model_metadata_sha256=hashes,cases=smoke_proof)
        receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(f'SMOKE PASSED: {config["reasoning_effort"]} reasoning, strict budget8/2048, max_new_tokens5120, math/code/multi, all endpoints',flush=True)
    if args.smoke_only:return
    for part in parts:run_part(config,tokenizer,endpoints,part,output_root,pin)

if __name__=='__main__':
    main()
