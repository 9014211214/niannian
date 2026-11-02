"""Explicit scripted plumbing responses, never real model/effect evidence."""
import json
from aria.runtime import MinimalModelClients


class Fixture:
    source_id = 'software_fixture'
    model = last_model = 'software_fixture_not_a_model'
    online_decision_version = 'online_turn_decision_v5_bound'
    online_return_format = 'json_schema_v1'
    reflection_return_format = 'body_json'
    memory_prefix_recovery = True
    reflection_itemwise_review = True
    last_finish_reason = 'stop'
    last_usage = {}
    last_diagnostic = {}
    timeout = 90

    def __init__(self, role, repeat):
        self.role=role; self.calls=0
        self.source=next(op['text'] for op in repeat['sessions'][0]['operations'] if op['op']=='say')

    def complete(self,messages,**requirements):
        self.calls+=1
        if self.role=='memory':
            rows=[]
            if self.calls==1:
                rows=[{'do':'聊天时先听我说完，一次最多问一个问题。','scope':self.source,
                    'quotes':[{'turn_index':0,'text':self.source}], 'confirmation':None,'supersedes':None,
                    'semantic_review':{'version':'preference_semantics_v1','kind':'communication_style',
                        'owner_requested':True,'quote_supported':True,'scope_preserved':True,'no_permission_claim':True}}]
            result={'stated_preferences':rows,'cognitive_privacy':[],'cognitive_observations':[],
                    'episodes':[],'relationships':[]}
        elif self.role=='reflection':
            result={'protocol_version':'2.0','reviews':[],'changes':[]}
        elif self.role=='safety_review':
            result={'conflicts':False,'delete_sentence_ids':[]}
        else:
            body='【软件替身】这是一条用于检查流程的固定回复。'
            result={'response':body,'state_signals':[],
                'current_affect':{'label':'neutral','confidence':0,'divergence':{'possible':False,'confidence':0,'pattern':''}},
                'selected_intervention_id':'','communication_plan':{'pace':'brief_exchange','support_order':'响应软件输入','boundary':'不执行外部动作'},
                'uncertainty':1,'safety_concern':'none',
                'safety_evidence':{'quote':'','subject':'not_applicable','time':'not_applicable','urgency':'not_applicable'},
                'applied_preference_ids':[]}
            if requirements.get('require_preference_control'):
                result['preference_control']={'action':'confirm'}
                result['response_segments']=[{'text':body,'claims_saved':False}]
        return json.dumps(result,ensure_ascii=False)


def fixture_clients(repeat):
    return MinimalModelClients(online=Fixture('online_fast',repeat),memory=Fixture('memory',repeat),
        reflection=Fixture('reflection',repeat),safety_review=Fixture('safety_review',repeat),
        model_calls_enabled=True,release_profile='release_v2')
