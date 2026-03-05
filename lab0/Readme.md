модель Галилея
$$
y = - \frac{g}{2v_{0}^{2}\cos^{2}\alpha}x^{2}+(\tan \alpha)x,\;0<\alpha< \frac{\pi}{2}
$$
модель Ньютона
$$
\begin{cases}
m \frac{du}{dt} = -\beta u \sqrt{ u^{2}+w^{2 }}\;,& \frac{dx}{dt}=u \\
m \frac{dw}{dt}=-g-\beta w\sqrt{  u^{2}+w^{2 }} \;, & \frac{dy}{dt}=w
\end{cases}
$$

Начальные условия: $x_{0}=y_{0}=0$
Скорости: сделать проекции на оси
$\alpha=45\degree ,\, v_{0}=1 \frac{м}{с}$
$\vec{F} = -\beta \vec{v}^2$
$\beta = \frac{cs\rho}{2}, \, C = 0,15$
масса = m

---

$$
\begin{align}
u=v_{0}\cos\alpha \\
w=v_{0}\sin\alpha \\ \\
\begin{cases}
y=(v_{0}\sin\alpha) t- \frac{gt^{2}}{2} \\
x=(v_{0}\cos\alpha )t
\end{cases}\\
t= \frac{x}{v_{0}\cos\alpha} \\
y= \frac{v_{0}\sin \alpha}{v_{0}\cos\alpha}x - \frac{g}{2} \frac{x^2}{v_{0}^{2}\cos^{2}\alpha} \\
y=- \frac{gx^{2}}{v_{0}^{2}\cos^{2}\alpha} + (\tan\alpha)x=0 \implies x=0 \\
\frac{gx}{v_{0}^{2}\cos^{2}\alpha} = (\tan\alpha) \\
x= \frac{2}{g}\tan \alpha v_{0}^{2}\cos^{2}\alpha \\
\end{align}
$$
$$
\begin{align}
m\vec{a} = m \vec{g} -\beta\vec{v}^{2} \\
\begin{cases}
\cancel m \frac{du}{dt} = 0-\frac{\beta u}{m} \sqrt{ u^{2} + w^{2} } \\
\cancel m \frac{dw}{dt} = \cancel m g - \frac{\beta w}{m} \sqrt{ u^{2}+w^{2} } \\
+ \text{начальные условия}
\end{cases} \\
(-\beta u\sqrt{ u^{2}+w^{2} })^{2}  + (-\beta w \sqrt{ u^{2}+w^{2} })^{2}=\beta^{2}(u^{2}+w^{2})(u^{2}+w^{2})=(\beta(u^{2}+w^{2}))^{2}=F_{c}^{2}
\end{align}
$$

---
