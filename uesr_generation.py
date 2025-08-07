import json
from openai import OpenAI
from sklearn.metrics import classification_report

# llm_client = OpenAI(
#                 api_key="sk-6s7A81DVbtPGCQfChn2EeJcIEk44jYiZZQCK9nyn94C96M3J",
#                 base_url="https://api.key77qiqi.cn/v1"
#             )

llm_client = OpenAI(
                api_key="xUfZX8zY3Te9hOg6U77pLB16g0b5AXs1S-7q9IOYJuI",
                base_url="https://zhenze-huhehaote.cmecloud.cn/inference-api/exp-api/inf-1336844822260682752/v1"
            )

label_dic={
    "真实":0,
    "虚假":1
}
en_label_dic={
    "real":0,
    "fake":1
}

def query_llm() -> str:
    """向LLM查询感染判定"""
    prompt = f"""
        你在社交媒体上一个用户,请任意生成一段你的个人描述,包括你性别,职业,年龄,受教育程度,性格以及爱好等等,模板:[女|医生|30岁|博士|活泼开朗|唱歌,跳舞].请按照上述模板,每个要素之间用|隔开,请注意尽可能的简短.
        """

    messages = [{"role": "system", "content": prompt}]

    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            temperature=1.2,
        )
        answer = response.choices[0].message.content.strip('[').strip(']').replace("\n","")
        print(answer)
        return answer
    except Exception as e:
        print(f"API调用失败: {e}")
        return None


def read_data():
    file_path = f"zh_temp.json"
    with open(file_path,"r",encoding="UTF-8") as r:
        results = json.load(r)
    return results

def calculate_metric(y_true,y_pred):
    report = classification_report(y_true, y_pred, target_names=["True", "False"])
    print(report)
    return report


from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    users = []
    samples = read_data()


    # 定义线程安全的处理函数
    def process_sample(id):
        # print(sample['label'])  # 注意：多线程打印可能乱序，建议移除或加锁
        user_describe = query_llm()
        return id,user_describe # 返回元组保证顺序一致性


    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=10) as executor:  # 调整max_workers控制线程数
        # 提交所有任务到线程池
        future_to_sample = {executor.submit(process_sample,id): id for id in range(1,100)}

        # 按提交顺序收集结果（保证y_true和y_pred顺序一致）
        for future in as_completed(future_to_sample):
            try:
                id,user_describe = future.result()  # 获取任务结果
                users.append({'id':id,'describe':user_describe})
            except Exception as e:
                print(f"Sample processing error: {e}")
    print(users)
    with open(f"users.json","w",encoding="UTF-8") as w:
        json.dump(users,w,indent=4,ensure_ascii=False)

