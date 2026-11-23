"""CPU verification using the deployed SGLang request and grammar classes."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from generate import fingerprint, input_rows, prompt_ids, request_payload, template_kwargs, validate_template
from launch import JOB, server_command
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from transformers import AutoTokenizer


def main():
    config=json.loads((JOB/'config.json').read_text())
    parser=argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args=parser.parse_args(server_command(config,0)[3:])
    assert args.model_path==config['model']
    assert args.enable_strict_thinking and args.reasoning_parser=='qwen3' and args.grammar_backend=='xgrammar'
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
    messages=[dict(role='user',content='test')]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs(config))
    assert rendered.endswith('<|im_start|>assistant\n<think>\n')
    validate_template(config,rendered)
    history=[*messages,dict(role='assistant',content='ANSWER_MARKER',reasoning_content='THOUGHT_MARKER'),
        dict(role='user',content='next question')]
    replay=tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=True,**template_kwargs(config))
    assert '<think>\nTHOUGHT_MARKER\n</think>\n\nANSWER_MARKER' in replay
    assert replay.count('THOUGHT_MARKER')==1
    generation=json.loads((Path(config['model'])/'generation_config.json').read_text())
    assert set(generation['eos_token_id'])==set(config['stop_token_ids'])

    payload=request_payload(config,messages,'preflight',0)
    request=ChatCompletionRequest.model_validate(payload)
    assert request.reasoning_effort==config['reasoning_effort']
    sampling_dict=request.to_sampling_params([],generation)
    params=SamplingParams(**sampling_dict)
    assert params.max_new_tokens==5120
    assert params.custom_params['thinking_budget']==2048
    for name in ['temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty']:
        assert getattr(params,name)==config[name],name

    # Exercise the scheduler's actual per-request budget assignment and the
    # exact ReasonerGrammarObject boundary used by strict thinking.
    end_ids=tokenizer.encode('</think>',add_special_tokens=False)
    assert len(end_ids)==1, 'This receipt expects the Qwen single-token close delimiter'
    filters=[]
    grammar=ReasonerGrammarObject(None,end_ids,enable_token_filter=True,
        token_filter_fn=lambda mask,ids,idx,allowed,**kw:filters.append((ids,allowed)))
    manager=object.__new__(GrammarManager)
    manager.scheduler=SimpleNamespace(model_config=SimpleNamespace())
    req=SimpleNamespace(grammar=grammar,sampling_params=params)
    manager._apply_request_reasoning_config(req)
    assert grammar.max_think_tokens==2048
    grammar.maybe_init_reasoning(True)
    token=tokenizer.encode('a',add_special_tokens=False)[0]
    assert token not in end_ids
    for _ in range(2047):grammar.accept_token(token)
    assert grammar._can_think_more()
    grammar.accept_token(token)
    assert not grammar._can_think_more()
    grammar.fill_vocab_mask(None,0)
    assert filters[-1]==(end_ids,True),filters
    grammar.accept_token(end_ids[0])
    assert grammar.tokens_after_end==0

    samples=[]
    for part in range(config['input_shards']):
        for index,row in enumerate(input_rows(config,part)):
            samples.append(dict(part=part,id=row['id'],user_turns=sum(m['role']=='user' for m in row['conversations']),
                first_prompt_tokens=len(prompt_ids(tokenizer,[row['conversations'][0]],config))))
            if index==4:break
    pin,hashes=fingerprint(config)
    receipt=dict(status='passed',gpu_tested=False,server_cli='passed',
        reasoning_effort=config['reasoning_effort'],native_template='passed',multi_turn_reasoning_replay='passed',
        request_to_sampling_params='passed',hard_budget_2048_boundary='passed',
        backend_max_new_tokens=params.max_new_tokens,backend_thinking_budget=params.custom_params['thinking_budget'],
        think_end_ids=end_ids,request_payload=payload,input_file=config['input_file'],
        input_bytes=Path(config['input_file']).stat().st_size,
        input_files=[dict(path=p,bytes=Path(p).stat().st_size) for p in config['input_files']],samples=samples,
        server_flags=server_command(config,0)[3:],fingerprint=pin,metadata_sha256=hashes,
        config_sha256=hashlib.sha256((JOB/'config.json').read_bytes()).hexdigest())
    (JOB/'preflight.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k in ['status','gpu_tested','request_to_sampling_params',
        'hard_budget_2048_boundary','backend_max_new_tokens','backend_thinking_budget','think_end_ids']}))

if __name__=='__main__':main()
