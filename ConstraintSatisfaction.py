import re
def solve(q):
    try:
        n=next(i for i in q if i.isalpha())
    except StopIteration:
        return q if eval(re.sub(r'(^|[^0-9])0+([1-9]+)',r'\1\2', q)) else False
    else:
        for i in (str(i) for i in range(10) if str(i) not in q):
            r=solve(q.replace(n,str(i)))
            if r:
                return r
        return False

query=input()
r=solve(query)
print(r) if r else print("No Solution Found")