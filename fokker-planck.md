# 1

We have a system of 5 reactions. Therefore we can write the chemical Langevin equation. The corresponding Fokker-Planck equation is:

$$
\frac{\partial P(x_1,x_2,x_3,t)}{\partial t}=-\sum_{i=1}^3\frac{\partial }{\partial x_i}[f_i(x_1,x_2,x_3)P(x_1,x_2,x_3,t)]+\frac{1}{2}\sum_{i=1}^3\sum_{k=1}^3\frac{\partial^2}{\partial x_i \partial x_k}[D_{ik}(x_1,x_2,x_3)P(x_1,x_2,x_3,t)]
$$

where the terms are given by:

$$
V=(v_{ij})=\begin{pmatrix}
1&-1&0&0&0\\
0&0&1&-1&0\\
0&0&0&1&-1\end{pmatrix}
$$

$$
a_f=0.5*(a_t-x_2-1+\sqrt{(a_t-x_2-1)^2+4a_t)}
$$

$$
s=A(1+\sin(\omega t))/c_d
$$

$$
g=\frac{(ef-1)s+1}{((f-1)s+1)((e-1)s+1)}
$$

$$
a_1=\alpha a_f/a_t\text{,  }a_2=x_1\text{,  }a_3=\alpha_0gx_1\text{,  }a_4=x_2\text{,  }a_5=x_3
$$

$$
f_i(x_1,x_2,x_3)=\sum_{j=1}^5v_{ij}a_j(x_1,x_2,x_3)
$$

$$
D_{ik}(x_1,x_2,x_3)=\sum_{j=1}^5v_{ij}v_{kj}a_j(x_1,x_2,x_3)
$$

and parameters are given:

$$
\alpha=200,a_t=40,\alpha_0=1,c_d=100,e=50,f=50,A=50,\omega=\pi
$$

The initial state is $x_{1,0}=20,x_{2,0}=70,x_{3,0}=70$ and the range of the variables is 0-100, 0-200, 0-200.









The chemical reactions in this system are:

- $$
  \emptyset \overset{\alpha a_f/a_t}\longrightarrow M
  $$
- $$
  M\overset{M}\longrightarrow \emptyset
  $$
- $$
  \emptyset \overset{\alpha_0gM}\longrightarrow P
  $$
- $$
  P\overset{P}\longrightarrow P_{nuc}
  $$
- $$
  P_{nuc}\overset{P_{nuc}}\longrightarrow\emptyset
  $$
