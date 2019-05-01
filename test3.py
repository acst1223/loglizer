import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime


x = ["03:33:12", "05:12:27", "10:34:55", "19:00:01", "21:12:44"]
y = [15, 85, 25, 63, 23]
x = [datetime.datetime.strptime(t, '%H:%M:%S') for t in x]

ax = plt.subplot(1, 1, 1)
plt.plot(x, y, '-o')
plt.xlabel('abc')
plt.ylabel('xyz')
ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
for label in ax.get_xticklabels():
    label.set_rotation(30)
plt.savefig("1.png")
plt.close()
