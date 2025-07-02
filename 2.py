import json


def convert_ccks_to_stat(input_file, output_file="../converted_train.json"):
    converted_data = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line.strip())
            text_id = entry["text_id"]
            text = entry["text"]

            # 将文本按句号拆分为句子
            sentences = [sent.strip() + "。" for sent in text.split("。") if sent.strip()]

            # 解析实体
            ann_valid_mspans = [attr["entity"] for attr in entry["attributes"]]

            # 生成事件信息
            event_dict = {}
            for attr in entry["attributes"]:
                if attr["type"] not in event_dict:
                    event_dict[attr["type"]] = attr["entity"]

            recguid_eventname_eventdict_list = [[f"event_{text_id}", entry["level3"], event_dict]]

            converted_data.append([
                text_id,
                {
                    "ann_valid_mspans": ann_valid_mspans,
                    "sentences": sentences,
                    "recguid_eventname_eventdict_list": recguid_eventname_eventdict_list
                }
            ])

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(converted_data, fout, ensure_ascii=False, indent=2)

    print(f"Converted data saved to {output_file}")


# 调用转换函数
convert_ccks_to_stat("ccks_task1_train.txt", "converted_train.json")
