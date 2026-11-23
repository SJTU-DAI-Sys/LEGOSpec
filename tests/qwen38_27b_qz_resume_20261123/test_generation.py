import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate as g
from split_input import split
from launch import launch_state,server_environment,server_command
from set_concurrency import migrate

CONFIG=json.loads((Path(__file__).parent/'config.json').read_text())

class Tokenizer:
    def encode(self,text,**kwargs):
        assert text=='</think>'
        return [99]

def response(thinking=2,finish='stop',final='The answer is 5.'):
    ids=[1]*thinking+[99,2,3]
    return dict(choices=[dict(message=dict(reasoning_content='short thought',content=final),finish_reason=finish)],
        sglext=dict(output_ids=[ids]),usage=dict(prompt_tokens=10,completion_tokens=len(ids)))

class GenerationTests(unittest.TestCase):
    def test_resume_prioritizes_unprocessed_and_still_retries_old_failures(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            job=Path(directory);out=job/'out';out.mkdir()
            source=job/'input.jsonl'
            source.write_text(''.join(json.dumps(dict(id=i,conversations=[dict(role='user',content=str(i))]))+'\n' for i in range(4)))
            config=dict(CONFIG,input_files=[str(source)],input_shards=1,concurrency_per_replica=1,fsync_every_records=1)
            existing=dict(id=0,status='success',conversations=[dict(role='assistant',final_content='old answer')])
            (out/'part0.jsonl').write_text(json.dumps(existing)+'\n')
            identity=dict(fingerprint='test',input=str(source),input_bytes=source.stat().st_size,input_mtime_ns=source.stat().st_mtime_ns,part=0,input_shards=1)
            (out/'part0.identity.json').write_text(json.dumps(identity))
            (out/'part0.truncated.jsonl').write_text(json.dumps(dict(id=1,status='truncated'))+'\n')
            (out/'part0.rejected.jsonl').write_text('')
            seen=[]
            def generate(config,tokenizer,endpoint,row):
                seen.append(row['id'])
                return dict(id=row['id'],status='success',conversations=[dict(role='assistant',final_content='new answer')])
            with patch.object(g,'generate_record',side_effect=generate):
                g.run_part(config,None,['local'],0,out,'test')
            self.assertEqual(seen,[2,3,1])
            rows=[json.loads(line) for line in (out/'part0.jsonl').read_text().splitlines()]
            self.assertEqual([row['id'] for row in rows],[0,2,3,1])
            self.assertEqual(rows[0],existing)

    def test_resume_does_not_repeat_expensive_generation_quality_check(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'answers.jsonl'
            path.write_text(json.dumps(dict(id=7,status='success',conversations=[dict(role='assistant',final_content='Accepted answer')]))+'\n'+ '{"partial"')
            done=set()
            with patch.object(g,'quality',side_effect=AssertionError('Resume must not rerun content heuristics')):
                handle=g.open_output(path,done);handle.close()
            self.assertEqual(done,{'7'})
            self.assertEqual(len(path.read_text().splitlines()),1)
            path.write_text(json.dumps(dict(id=7,status='success',conversations=[dict(role='assistant',final_content='')]))+'\n')
            with self.assertRaisesRegex(ValueError,'empty final answer'):g.open_output(path,set())

    def test_concurrency_and_kv_budget_migrate_without_changing_saved_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            job=Path(directory);out=job/'answers'/'default';out.mkdir(parents=True)
            config=dict(CONFIG,output_dir=str(job/'answers'),input_files=[],input_shards=2,replicas=4,concurrency_per_replica=128)
            config.pop('mem_fraction_static',None)
            saved={}
            fp=lambda c:(str(c['concurrency_per_replica'])+'/'+str(c.get('mem_fraction_static',.85)),{})
            for part in range(2):
                source=job/f'input{part}.jsonl';source.write_text('{}\n');config['input_files'].append(str(source))
                identity=dict(fingerprint=fp(config)[0],input=str(source),input_bytes=source.stat().st_size,input_mtime_ns=source.stat().st_mtime_ns)
                (out/f'part{part}.identity.json').write_text(json.dumps(identity))
                data=out/f'part{part}.jsonl';data.write_text('{"id":1}\n');saved[data]=data.read_bytes()
            (job/'config.json').write_text(json.dumps(config))
            receipt=migrate(job,256,fp,mem_fraction_static=.90)
            updated=json.loads((job/'config.json').read_text())
            self.assertEqual(receipt['changed_keys'],['concurrency_per_replica','mem_fraction_static'])
            self.assertEqual(receipt['total_concurrency'],2048)
            self.assertEqual(updated,dict(config,concurrency_per_replica=256,mem_fraction_static=.9))
            command=server_command(updated,0)
            for flag,value in [('--max-running-requests','256'),('--max-mamba-cache-size','256'),('--mem-fraction-static','0.9'),('--kv-cache-dtype','bfloat16')]:
                self.assertEqual(command[command.index(flag)+1],value)
            for data,value in saved.items():self.assertEqual(data.read_bytes(),value)

    def test_native_effort_markers(self):
        prefix='<|im_start|>assistant\n<think>\n'
        for effort in ('low','medium','xhigh'):
            rendered=('' if effort=='medium' else f'Reasoning effort is set to {effort}.')+prefix
            g.validate_template(dict(CONFIG,reasoning_effort=effort),rendered)
        with self.assertRaises(AssertionError):
            g.validate_template(dict(CONFIG,reasoning_effort='medium'),'Reasoning effort is set to low.'+prefix)

    def test_concurrency_resume_preserves_answers_and_rejects_unrelated_changes(self):
        for mismatch in [False,True]:
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                job=Path(directory);out=job/'answers'/'default';out.mkdir(parents=True)
                config=dict(CONFIG,output_dir=str(job/'answers'),input_files=[],input_shards=2,replicas=4,concurrency_per_replica=8)
                saved={}
                for part in range(2):
                    source=job/f'input{part}.jsonl';source.write_text('{}\n');config['input_files'].append(str(source))
                    identity=dict(fingerprint='wrong' if mismatch and part==1 else 'fp8',input=str(source),
                        input_bytes=source.stat().st_size,input_mtime_ns=source.stat().st_mtime_ns)
                    (out/f'part{part}.identity.json').write_text(json.dumps(identity))
                    data=out/f'part{part}.jsonl';data.write_text('{"id":1}\n');saved[data]=data.read_bytes()
                (job/'config.json').write_text(json.dumps(config))
                fp=lambda c:('fp'+str(c['concurrency_per_replica']),{})
                if mismatch:
                    with self.assertRaisesRegex(ValueError,'refusing migration'):migrate(job,64,fp)
                    self.assertEqual(json.loads((job/'config.json').read_text())['concurrency_per_replica'],8)
                    self.assertEqual(json.loads((out/'part0.identity.json').read_text())['fingerprint'],'fp8')
                else:
                    result=migrate(job,64,fp)
                    self.assertEqual(result['total_concurrency'],512)
                    self.assertEqual(json.loads((out/'part0.identity.json').read_text())['fingerprint'],'fp64')
                for data,value in saved.items():self.assertEqual(data.read_bytes(),value)

    def test_independent_hosts_cannot_share_build_metadata(self):
        first=server_environment(CONFIG,'worker0')
        second=server_environment(CONFIG,'worker1')
        self.assertNotEqual(first['FLASHINFER_WORKSPACE_BASE'],second['FLASHINFER_WORKSPACE_BASE'])
        self.assertEqual(first['SGLANG_CACHE_DIR'],first['FLASHINFER_WORKSPACE_BASE'])
        self.assertEqual(first['SGLANG_MAX_THINK_TOKENS'],'2048')
        self.assertEqual(server_environment(CONFIG,'worker0'),first)

    def test_startup_and_failure_state_keep_commands_separate_from_pids(self):
        plan=dict(host='gpu',parts=[0],servers=[['python','-m','sglang.launch_server']])
        for status,extra in [('starting_servers',{}),('failed',dict(error='server exited'))]:
            state=launch_state(plan,status,10,[11,12],None,**extra)
            self.assertEqual(state['servers'],plan['servers'])
            self.assertEqual(state['server_pids'],[11,12])
            self.assertEqual(state['status'],status)
            self.assertEqual(json.loads(json.dumps(state)),state)

    def test_budget_request_and_response(self):
        payload=g.request_payload(CONFIG,[dict(role='user',content='question')],'test',0)
        self.assertEqual(payload['max_tokens'],5120)
        self.assertEqual(payload['custom_params'],{'thinking_budget':2048})
        self.assertEqual(payload['reasoning_effort'],CONFIG['reasoning_effort'])
        assistant,detail=g.parse_response(CONFIG,Tokenizer(),'local',response(2048),payload,10)
        self.assertEqual(detail['thinking_tokens_before_close'],2048)
        with self.assertRaisesRegex(g.Rejected,'thinking_budget_violated'):
            g.parse_response(CONFIG,Tokenizer(),'local',response(2049),payload,10)

    def test_missing_ids_and_empty_or_truncated_final(self):
        payload=g.request_payload(CONFIG,[],'x',0)
        bad=response();del bad['sglext']
        with self.assertRaisesRegex(g.Rejected,'invalid_server_response'):
            g.parse_response(CONFIG,Tokenizer(),'local',bad,payload,10)
        for bad,reason in [(response(final=''),'empty_response'),(response(finish='length'),'truncated')]:
            with self.assertRaisesRegex(g.Rejected,reason):
                g.parse_response(CONFIG,Tokenizer(),'local',bad,payload,10)

    def test_multiturn_uses_new_history_once(self):
        row=dict(id=1,conversations=[dict(role='system',content='system'),dict(role='user',content='first'),
            dict(role='assistant',content='old answer'),dict(role='user',content='second')])
        calls=[]
        def complete(config,tokenizer,endpoint,messages,*args):
            calls.append(copy.deepcopy(messages))
            return dict(role='assistant',content=f'answer{len(calls)}',reasoning_content='thought'),{}
        with patch.object(g,'completion',side_effect=complete):
            result=g.generate_record(CONFIG,None,'local',row)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][2],dict(role='assistant',content='answer1',reasoning_content='thought'))
        saved=result['conversations'][2]
        self.assertEqual(saved['content'],'answer1')
        self.assertEqual(saved['reasoning_content'],'thought')
        self.assertEqual(saved['full_content'],'<think>\nthought\n</think>\n\nanswer1')
        self.assertEqual(len(result['generation']['turns']),2)

    def test_shards_cover_input_and_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            config=dict(CONFIG,input_file=str(Path(directory)/'input.jsonl'))
            Path(config['input_file']).write_text(''.join(json.dumps(dict(id=i,conversations=[
                dict(role='user',content='first'),dict(role='user',content='second')]))+'\n' for i in range(7)))
            split_dir=Path(directory)/'split'
            split(config['input_file'],split_dir)
            config['input_files']=[str(split_dir/f'part{i}.jsonl') for i in range(2)]
            shards=[[r['id'] for r in g.input_rows(config,p)] for p in range(2)]
            self.assertEqual(shards,[[0,2,4,6],[1,3,5]])
            self.assertTrue(all(len(row['conversations'])==2 for p in range(2) for row in g.input_rows(config,p)))
            path=Path(directory)/'output.jsonl'
            path.write_text(json.dumps(dict(id=4,status='success',conversations=[dict(role='assistant',
                content='<think>\nthought\n</think>\n\nanswer',final_content='answer')]))+'\n'+ '{"partial"')
            done=set();handle=g.open_output(path,done);handle.close()
            self.assertEqual(done,{'4'})
            self.assertEqual(len(path.read_text().splitlines()),1)

if __name__=='__main__':unittest.main(verbosity=2)
