import numpy as np
import matplotlib.pyplot as plt


def f1(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

def f2(t):
    return np.sin(2*np.pi*t)*np.cos(3*np.pi*t)

t = np.arange(0.0,5.0,0.02)

plt.figure(figsize=(8,7),dpi=98)
p1 = plt.subplot(211)
p2 = plt.subplot(212)

p1.plot(t,f1(t),"g-",label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
p2.plot(t,f2(t),"r-.",label="$g(t)=\sin (2 \pi t) \cos (3 \pi t)$",linewidth=2)

p1.axis([0.0,5.01,-1.0,1.5])

p1.set_ylabel("v",fontsize=14)
p1.set_title("A simple example",fontsize=18)
p1.grid(True)
p1.legend()

p2.axis([0.0,5.01,-1.0,1.5])
p2.set_ylabel("v",fontsize=14)
p2.set_xlabel("t",fontsize=14)
p2.legend()

plt.show()