# -*- coding:utf-8 -*-

def format_result(result, text, tag): 
    entities = [] 
    for i in result: 
        begin, end = i 
        entities.append({ 
            "start":begin, 
            "stop":end + 1, 
            "word":text[begin:end+1],
            "type":tag
        }) 
    return entities

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B_" + tag)
    mid_tag = tag_map.get("M_" + tag)
    end_tag = tag_map.get("E_" + tag)
    # single_tag = tag_map.get("S")
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags

def f1_score(tar_path, pre_path, tag, tag_map):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        # print(tar)
        # print(pre)
        tar_tags = get_tags(tar, tag, tag_map)
        # print(tar_tags)
        pre_tags = get_tags(pre, tag, tag_map)
        # print(pre_tags)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
    return recall, precision, f1