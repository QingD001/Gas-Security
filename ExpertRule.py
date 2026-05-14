import numpy as np
from typing import Tuple, Dict, List, Optional

def tofloat(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

#获取报警前后的序列
def get_prevals(row, cols: List[int]) -> List[float]:
    vals = []
    for c in cols:
        x = tofloat(row.iloc[c])
        if x is not None:
            vals.append(x)
    return vals


def get_postvals(row, cols: List[int]) -> List[float]:
    vals = []
    for c in cols:
        x = tofloat(row.iloc[c])
        if x is not None:
            vals.append(x)
    return vals

def r1(prevals: List[float], postvals: List[float], alarmval: float) -> Tuple[bool,str]:
    eps = 1e-3
    vals = prevals + [alarmval] + postvals
    for i in vals:
        if 0 < i < eps:
            return True, "出现异常小量"
    return False, "符合规则 R1"

def r2(threshold: float) -> Tuple[bool,str]:
    if threshold < 0 or 0 < threshold < 1e-2:
        return True, "阈值异常"
    return False, "符合规则 R2"

def r3(prevals: List[float], alarmval: float, threshold: float) -> Tuple[bool,str]:
    if alarmval <= 0 or len(prevals)<5 or threshold <= 0:
        return False, "符合规则 R3"
    cnt = 0
    while cnt<5:
        if prevals[-1-cnt]!=0:
            return False, "符合规则 R3"
        cnt+=1
    return True, "连续5次报警前值为0"

def r4(prevals: List[float], postvals: List[float], alarmval: float, threshold: float) -> Tuple[bool,str]:
    if alarmval <= 0 or len(prevals)<5 or len(postvals)<5 or threshold <= 0:
        return False, "符合规则 R4"
    cnt = 0
    while cnt<5:
        if prevals[-1-cnt]!=0 or postvals[cnt]!=0:
            return False, "符合规则 R4"
        cnt+=1
    return True, "连续5次报警前后值为0"

def r5(postvals: List[float]) -> Tuple[bool,str]:
    eps = 1e-3
    cnt = 0
    for v in postvals:
        if 0 < v < eps:
            cnt+=1
    if cnt>=len(postvals)/2:
        return True, "报警后异常小量占比过高"
    return False, "符合规则 R5" 

def judge_with_rules(prevals: List[float], postvals: List[float], alarmval: float, threshold: float) -> Dict[str, Tuple[bool,str]]:
    results = {}
    results['R1'] = r1(prevals, postvals, alarmval)
    results['R2'] = r2(threshold)
    results['R3'] = r3(prevals, alarmval, threshold)
    results['R4'] = r4(prevals, postvals, alarmval, threshold)
    results['R5'] = r5(postvals)
    return results