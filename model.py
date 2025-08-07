import networkx as nx
import numpy as np
import random
import re
import json
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
import json
from sklearn.metrics import classification_report

# 全局配置参数
N_AGENTS = 50
INIT_INFECTED = 1
MALICIOUS_RATIO = 0.05
CONSENSUS_THRESHOLD = 0.7
TIME_DECAY_FACTOR = 0.95

label_dict={
    "real":0,
    "fake":1
}

label_dic = {
    "real": 0,
    "fake": 1

}

class NewsEvent:
    """新闻事件类"""

    def __init__(self,news:dict):
        self.id = news['source_id']
        self.content = news['content']
        self.external_knowledge = f"写作表达:[{news['td_rationale']}],经验常识:[{news['cs_rationale']}]."
        self.verification_score = 0.5
        self.label = news['label']
        self.is_rumor = None
        self.predict = None
        self.creation_time = datetime.now()
        self.last_updated = datetime.now()
        self.propagations = 0
        self.state_history = defaultdict(list)

    def update_knowledge(self, new_evidence):
        self.external_knowledge = new_evidence
        self.last_updated = datetime.now()


class LLM_Simulator:
    """LLM状态转换引擎"""

    def __init__(self):
        self.templates = {
            # 无知者(S)的转换逻辑
            'S': {
                'I': lambda n, e, u: f"""You are a user on social media and you are seeing someone spreading a news story as well as their opinion:
                【news】{n}
                【viewpoints】{u}
                Please choose your standpoint:
                - I：[The news is real.]
                - R：[The news is fake]
                return JSON：{{"state": "I/R", "reason": "viewpoints for choosing this standpoint"}}"""
                ,
                'R': lambda n, e, u: f"""You are a user on social media and you are seeing someone spreading an opinion about a news story questioning the veracity of the news:
                【news】{n}
                【viewpoints】{u}
                Please choose your standpoint:
                - I：[The news is real.]
                - R：[The news is fake]
                return JSON：{{"state": "I/R", "reason": "viewpoints for choosing this standpoint"}}"""
            },
            # 中立者(N)的转换逻辑
            'N': {
                'external': lambda n, e, u: f"""You need to combine external knowledge to determine the truthfulness of the news, please note that external knowledge may not be completely accurate:
                【news】{n}
                【knowledge】{e}
                Please choose your standpoint:
                - I：[The news is real.]
                - R：[The news is fake]
                return JSON：{{"state": "I/R", "reason": "viewpoints for choosing this standpoint"}}"""
            },
            # 理智者(R)的转换逻辑
            'I': {
                'R': lambda n, e, u: f"""You are spreading a news story that you believe to be true, and you hear the opinion of someone who questions the veracity of the news:
                【viewpoints】{u}
                【news】{n}
                【knowledge】{e}
                Please choose your standpoint:
                - I：[The news is real.]
                - R：[The news is fake]
                return JSON：{{"state": "I/R", "reason": "viewpoints for choosing this standpoint"}}"""
            },
            # 理智者(R)的转换逻辑
            'R': {
                'I': lambda n, e, u: f"""You are questioning the veracity of a news story and receive an opinion that someone thinks the story is news:
                       【viewpoints】{u}
                       【news】{n}
                       【knowledge】{e}
                       Please choose your standpoint:
                        - I：[The news is real.]
                        - R：[The news is fake]
                        return JSON：{{"state": "I/R", "reason": "viewpoints for choosing this standpoint"}}"""
            }
        }

    def query(self, receiver_state, sender_state, news_content, external_knowledge="", user_view=""):

        while(True):
            try:
                # 中立者特殊处理外部知识
                if receiver_state == 'N':
                    template = self.templates['N']['external']
                else:
                    template = self.templates[receiver_state][sender_state]

                prompt = template(news_content, external_knowledge, user_view)
                # if receiver_state == 'N':
                #     print(prompt)
                response = llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=150
                )

                result = json.loads(
                    response.choices[0].message.content.strip("```").strip("json").strip("```").strip("json").strip("\n"))
                # if receiver_state == 'N':
                # print(response.choices[0].message.content.strip("```").strip("json").strip("```").strip("json").strip(
                #     "\n"))
                if result.get("state")!=None:
                    break
            except Exception as e:
                print(f"API错误: {str(e)}")
                return {"state": receiver_state}  # 失败时保持原状态
        return result


class Agent:
    """网络节点代理"""

    def __init__(self, agent_id):
        self.id = agent_id
        self.state = 'S'
        self.llm = LLM_Simulator()
        self.prev_state = 'S'  # 新增：记录前一个状态
        self.interaction_log = []
        self._init_default_log()
        self.debug = False
        self.reason = ""

    def _init_default_log(self):
        self.interaction_log.append({
            'timestamp': datetime.now(),
            'previous_state': 'S',
            'response': {'state': 'S', 'reason': '初始状态'},
            'contact': None
        })

    def process_interaction(self, sender, news):
        """处理信息交互（精确传播路径控制）"""
        valid_paths = {
            'S': ['I', 'R'],  # S只能接收I和R的传播
            'N': ['external'],  # N只处理外部知识
            'R': ['I'],  # R只能接收I的传播
            "I": ["R"]
        }

        # 验证传播路径合法性
        if sender and sender.state not in valid_paths[self.state]:
            return self.state

        # 获取传播者观点
        sender_view = None
        if sender and len(sender.interaction_log) > 0:
            last_log = sender.interaction_log[-1]
            sender_view = f"{last_log['response'].get('reason', '')} [来自用户{sender.id}]"

        # LLM状态转换
        response = self.llm.query(
            self.state,
            sender.state if sender else 'external',
            news.content,
            news.external_knowledge,
            sender_view
        )

        # 记录交互日志
        log_entry = {
            'timestamp': datetime.now(),
            'previous_state': self.state,
            'response': response,
            'contact': sender.id if sender else None
        }

        self.interaction_log.append(log_entry)
        # print(response)
        new_state = response.get('state', self.state)
        new_reason = response.get('reason',self.reason)

        if self.state == 'N' and sender is None:  # 外部知识触发
            if new_state not in ['I', 'R']:
                new_state = random.choice(['I', 'R'])  # 强制转换
                print(f"Agent {self.id} (N) 未选择有效状态，强制转为 {new_state}")

        # 更新状态
        if new_state in ['S', 'N', 'I', 'R']:
            self.prev_state = self.state  # 在更新前保存旧状态
            self.state = new_state
            self.reason = new_reason

        return self.state

    def has_state_changed(self):
        """检查状态是否发生变化"""
        return self.state != self.prev_state

class MALPModel:
    """完整传播模型"""

    def __init__(self, newsEvent:NewsEvent,test_mode=False):
        self.G = nx.barabasi_albert_graph(N_AGENTS, 3)
        self.agents = {n: Agent(n) for n in self.G.nodes()}
        self.news = newsEvent
        self._init_network()
        self.time_step = 0
        self.test_mode = test_mode
        self.changed_nodes = set()  # 新增：记录状态变化的节点

    def _init_network(self):
        # 初始化传播者
        init_spreader = random.choice(list(self.G.nodes()))
        self.agents[init_spreader].state = 'I'
        self.agents[init_spreader].interaction_log.append({
            'timestamp': datetime.now(),
            'previous_state': 'S',
            'response': {'state': 'I', 'reason': '初始传播源'},
            'contact': None
        })

        # 初始化恶意节点
        # mal_nodes = random.sample(self.G.nodes(), int(N_AGENTS * MALICIOUS_RATIO))
        # for n in mal_nodes:
        #     self.agents[n].state = 'I'
        #     self.agents[n].influence = 2.0

    def _update_global_state(self):
        states = [a.state for a in self.agents.values()]

        self.news.state_history['S'].append(states.count('S'))
        self.news.state_history['N'].append(states.count('N'))
        self.news.state_history['I'].append(states.count('I'))
        self.news.state_history['R'].append(states.count('R'))



    def _get_targets(self, spreader_state):
        """获取合法传播目标"""
        target_map = {
            'I': ['S', 'R'],  # 信谣者传播给无知者和辟谣者
            'R': ['I'],  # 辟谣者只说服信谣者
            'N': []  # 中立者不主动传播
        }
        return target_map[spreader_state]

    def run_iteration(self):

        self.time_step += 1
        should_continue = True
        self.changed_nodes.clear()  # 清空上一轮变化记录

        # --- 单阶段传播流程 ---
        # 步骤1: 收集当前活跃传播者
        active_spreaders = [
            n for n in self.G.nodes()
            if self.agents[n].state in ['I', 'R']
        ]

        temp_changed = []

        # 处理中立者(N)的状态转换
        neutrals = [n for n in self.G.nodes() if self.agents[n].state == 'N']
        for node in neutrals:
            new_state = self.agents[node].process_interaction(None, self.news)

        # 步骤2: 处理传播并记录状态变化
        for node in active_spreaders:

            if self.agents[node].state == self.agents[node].prev_state:
                continue
            # print(f"传播节点：{node}")
            agent = self.agents[node]
            neighbors = list(self.G.neighbors(node))
            valid_targets = self._get_targets(agent.state)

            for neighbor in neighbors:
                target = self.agents[neighbor]
                if target.state not in valid_targets:
                    continue

                prev_state = target.state
                new_state = target.process_interaction(agent, self.news)

                # 记录目标节点的状态变化
                if new_state != prev_state:
                    temp_changed.append(neighbor)
                    if new_state == 'I':
                        self.news.propagations += 1
                    elif new_state == 'R' and prev_state == 'I':
                        self.news.propagations = max(0, self.news.propagations - 1)

            self.agents[node].prev_state = self.agents[node].state

        self._update_global_state()
        if len(temp_changed)==0:
            for id in range(0,50):
                self.agents[id].prev_state = "S"

        self.Print_agent_statt()

        # 终止条件判断
        current_states = set(a.state for a in self.agents.values())
        if current_states <= {'S', 'I'} or current_states <= {'S', 'R'}:
            should_continue = False
        states = [a.state for a in self.agents.values()]
        if states.count('I') > states.count('R'):
            self.news.predict = "real"
        else:
            self.news.predict = "fake"
        model.visualize()
        return should_continue

    def Print_agent_statt(self):
        for id in range(0,len(self.agents)):
            print(f"用户:{id+1},状态:{self.agents[id].state},原因:{self.agents[id].reason}")

    def visualize(self):
        current = {k: v[-1] for k, v in self.news.state_history.items()}
        # print(f"\n=== Step {self.time_step} ===")
        print(f"无知者(S): {current['S']} | 中立者(N): {current['N']}")
        print(f"信谣者(I): {current['I']} | 辟谣者(R): {current['R']}")


# 运行示例
if __name__ == "__main__":
    file_name = "en_test_result_NER.json"

    with open(file_name,"r",encoding="UTF-8") as r:
        resluts = json.load(r)


    for id,news in enumerate(resluts,1):


        newsEvent = NewsEvent(news)
        model = MALPModel(newsEvent)
        print("------------------------------------------------")
        print(f"第{id}条新闻")
        print(f"新闻内容：{model.news.content}")
        print(f"新闻标签：{news['label']}")

        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            print(f"\n=== 迭代 {step + 1}/{MAX_STEPS} ===")
            if not model.run_iteration():
                break

        print("\n=== 最终报告 ===")
        print(f"新闻ID: {model.news.id}")
        print(f"新闻标签:{model.news.label}")
        print(f"新闻预测:{model.news.predict}")
        print(f"最终状态分布: S-{model.news.state_history['S'][-1]} N-{model.news.state_history['N'][-1]} I-{model.news.state_history['I'][-1]} R-{model.news.state_history['R'][-1]}")
        print("\n")
