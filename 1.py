# Data/build_data.py
import json


event_type_map = {
    '兼职刷单类': '0',
    '冒充身份类': '1',
    '用户有泄露信息': '2',
    '其他亲属/其他朋友': '3',
    '广义欺诈类': '4',
    '不发货类': '5',
    '用户泄露信息': '6',
    '伴侣': '7',
    '微信社交负面': '8',
    '设备丢失': '9',
    '违规违禁类': '10',
    '其他': '11',
    '亲友操作': '12',
    '收到商品/服务与约定不符合': '13',
    '直系亲属': '14',
    '欺诈风险': '15',
    '商户风险': '16',
    '胁迫': '17',
    '好友/领导/婚恋交友': '18',
    '无效内容': '19',
    '投资理财/高额返利': '20',
    '信息未泄露': '21',
    '付款后未收到商品|服务': '22',
    '灰产推广（借呗/花呗提额推广/赌博网站等）': '23',
    '色情/赌博/彩票': '24',
    '客服': '25',
    '盗用风险': '26',
    '刷单类': '27'
}


attributes_type_map = {
    '案发城市': 'afcs',
    '受害人': 'shr',
    '受害人身份': 'shrsf',
    '身份证号': 'sfzh',
    '嫌疑人': 'xyr',
    '案发时间': 'afsj',
    '资损金额': 'zsje',
    '涉案平台': 'sapt',
    '银行卡号': 'yhkh',
    '支付渠道': 'zfqd',
    '订单号': 'ddh',
    '手机号': 'sjh',
    '交易号': 'jyh'
}
def convert_to_dee_format(input_data):
    dee_format = []
    for item in input_data:
        doc_item = {
            "id": item["ID"],
            "content": item["Doc"],
            "events": []
        }

        if "Event" in item and item["Event"]:
            event = {
                "type": item["Event"]["Type"],
                "arguments": {}
            }

            # 转换论元
            if "Arguments" in item["Event"]:
                for arg_name, arg_value in item["Event"]["Arguments"].items():
                    if arg_name in attributes_type_map:
                        event["arguments"][attributes_type_map[arg_name]] = arg_value

            doc_item["events"].append(event)

        dee_format.append(doc_item)

    return dee_format


# 读取原始数据
with open("D:/event_extraction_model/DocEE-main/DocEE-main/ccks_task1_train_dcfee.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 转换格式
dee_data = convert_to_dee_format(raw_data)

# 保存转换后的数据
with open("E:/typed_train.json", "w", encoding="utf-8") as f:
    json.dump(dee_data, f, ensure_ascii=False, indent=2)