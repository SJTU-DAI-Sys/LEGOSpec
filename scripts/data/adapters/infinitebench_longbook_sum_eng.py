"""InfiniteBench longbook_sum_eng: complete context followed by its input."""
from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    context=row['context']
    question=row['input']
    if not isinstance(context,str) or not context.strip():
        raise ValueError('InfiniteBench context must be nonempty text')
    if not isinstance(question,str) or not question.strip():
        raise ValueError('InfiniteBench input must be nonempty text')
    return {'prompt':context+'\n\n'+question,
            'source':'xinrongzhang2022/InfiniteBench','category':'longbook_sum_eng'}
