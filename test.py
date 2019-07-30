from datetime import datetime
import time

ltime = time.localtime(1563808497)
a = time.strftime('%Y-%m-%dT%H:%M:%S', ltime)
print(a)