"""
Aplicações Reais de Derivadas em Computação (5 problemas)
- Problem 1: Otimização de tempo de resposta / alocação de recursos
- Problem 2: Treinamento de modelo via Gradient Descent (regressão simples)
- Problem 3: Análise de adoção (curva logística) e ponto de inflexão
- Problem 4: Detecção de bordas em imagem sintética (Sobel)
- Problem 5: EOQ aplicado a estoque de hardware (LEC) e análise de sensibilidade

Dependências:
  pip install numpy matplotlib sympy opencv-python scipy
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import cv2
from math import sqrt
from typing import Tuple

# ---------------------------
# PROBLEMA 1 — Otimização de tempo de resposta (TI)
# Modelo realista: tempo médio por requisição T(x) = A / x + B * x + C
# A representa carga/throughput proporcional, B representa overhead por recurso.
# ---------------------------
def problema1_otimiza_servidor(A=1200.0, B=1.8, C=0.0):
    """
    Encontra a alocação ótima de recursos x>0 que minimiza T(x) = A/x + B*x + C.
    Plota a curva e marca o mínimo.
    """
    x = sp.symbols('x', positive=True)
    T = A / x + B * x + C
    T_prime = sp.diff(T, x)
    sol = sp.solve(T_prime, x)
    # Filtrar solução real positiva
    sol_real = [s for s in sol if s.is_real and float(s) > 0]
    if not sol_real:
        raise RuntimeError("Sem solução positiva")
    x_opt = float(sol_real[0])
    T_opt = float(T.subs(x, x_opt))

    print("Problema 1 - Otimização de Servidor")
    print(f"Modelo: T(x) = {A}/x + {B}*x + {C}")
    print(f"x ótimo = {x_opt:.3f} recursos -> T = {T_opt:.3f}")

    # Plot
    x_vals = np.linspace(max(0.1, x_opt * 0.2), x_opt * 3.0, 400)
    T_fun = sp.lambdify(x, T, 'numpy')
    plt.figure(figsize=(8,4))
    plt.plot(x_vals, T_fun(x_vals), label='T(x)')
    plt.scatter([x_opt], [T_opt], color='red', zorder=5, label=f'Ótimo x={x_opt:.2f}')
    plt.xlabel('Recursos (x)')
    plt.ylabel('Tempo médio por requisição T(x)')
    plt.title('Problema 1 — Otimização de Alocação de Recursos (TI)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_opt, T_opt

# ---------------------------
# PROBLEMA 2 — Gradient Descent aplicável a ML (regressão linear simples)
# Geramos dados sintéticos (y = m*x + b + ruído) e ajustamos com GD.
# ---------------------------
def problema2_gradient_descent(learning_rate=0.01, iters=2000, seed=0):
    """
    Treina uma regressão linear simples y = w*x + b por Gradient Descent.
    Gera gráfico da convergência e compara com solução analítica (OLS).
    """
    np.random.seed(seed)
    # dados sintéticos
    N = 120
    x = np.linspace(0, 10, N)
    true_w, true_b = 2.5, -1.0
    y = true_w * x + true_b + np.random.normal(scale=1.0, size=N)

    # inicialização
    w, b = 0.0, 0.0
    history = []
    for i in range(iters):
        # predição
        y_pred = w * x + b
        # gradientes MSE
        dw = (2.0 / N) * np.dot(y_pred - y, x)
        db = (2.0 / N) * np.sum(y_pred - y)
        # atualização
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % (iters // 50 + 1) == 0:
            loss = np.mean((y_pred - y) ** 2)
            history.append((i, loss, w, b))

    # solução analítica (OLS)
    X = np.vstack([x, np.ones_like(x)]).T
    ols_params = np.linalg.lstsq(X, y, rcond=None)[0]
    ols_w, ols_b = ols_params[0], ols_params[1]

    print("Problema 2 - Gradient Descent (Regressão Linear Simples)")
    print(f"Parâmetros verdadeiros: w={true_w}, b={true_b}")
    print(f"Parâmetros GD ao final: w={w:.4f}, b={b:.4f}")
    print(f"Parâmetros OLS: w={ols_w:.4f}, b={ols_b:.4f}")

    # Plots: dados e ajuste
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(x, y, s=15, label='Dados')
    plt.plot(x, true_w*x + true_b, label='Verdadeiro', linewidth=2)
    plt.plot(x, w*x + b, label='Ajuste (GD)', linewidth=2)
    plt.plot(x, ols_w*x + ols_b, '--', label='Ajuste (OLS)', linewidth=2)
    plt.title('Ajuste por Gradient Descent vs OLS')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)

    # Plots: perda
    plt.subplot(1,2,2)
    iters_hist = [h[0] for h in history]
    loss_hist = [h[1] for h in history]
    plt.plot(iters_hist, loss_hist, '-o')
    plt.title('Convergência (MSE)')
    plt.xlabel('Iteração'); plt.ylabel('MSE'); plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (w, b), (ols_w, ols_b)

# ---------------------------
# PROBLEMA 3 — Curva de adoção (logística) e Ponto de Inflexão
# f(t) = K / (1 + exp(-r*(t - t0))) -> inflexão em t = t0
# Geramos dados sintéticos e estimamos t0 com transformação logit simplificada.
# ---------------------------
def problema3_adocao_estimativa(K=10000, r=1.0, t0=6.0, noise_scale=200.0, seed=1):
    """
    Gera dados de adoção com um modelo logístico, adiciona ruído e estima o ponto
    de inflexão (t0). Mostra gráfico e interpretações práticas.
    """
    np.random.seed(seed)
    t = np.linspace(0, 12, 200)
    def logistic(t): return K / (1 + np.exp(-r * (t - t0)))
    adop = logistic(t) + np.random.normal(scale=noise_scale, size=t.shape)

    # Estimativa robusta do t0: pelo ponto onde f''(t)=0 (no modelo exato t0),
    # mas com dados ruidosos podemos aproximar: local máximo da derivada (f')
    dt = t[1] - t[0]
    f_prime = np.gradient(adop, dt)
    idx_max = np.argmax(f_prime)
    t_inflex_est = t[idx_max]

    print("Problema 3 - Adoção de Produto (Curva Logística)")
    print(f"Ponto de inflexão real (t0) = {t0}")
    print(f"Ponto de inflexão estimado pela derivada (máx f') = {t_inflex_est:.3f}")

    # Plots
    plt.figure(figsize=(10,6))
    plt.plot(t, adop, label='Dados (ruidosos)')
    plt.plot(t, logistic(t), '--', label='Logística (verdadeira)')
    plt.plot(t, np.gradient(adop, dt), label="Estimativa f'(t) (derivada numérica)")
    plt.axvline(t0, color='green', linestyle='--', label=f't0 real = {t0}')
    plt.axvline(t_inflex_est, color='red', linestyle=':', label=f'estimado = {t_inflex_est:.2f}')
    plt.title('Problema 3 — Curva de Adoção e Ponto de Inflexão')
    plt.xlabel('Tempo (meses)')
    plt.legend(); plt.grid(True)
    plt.show()

    return t_inflex_est

# ---------------------------
# PROBLEMA 4 — Detecção de bordas com Sobel (imagem sintética)
# Gera uma imagem com formas reais (retângulos, círculos) e aplica Sobel.
# ---------------------------
def problema4_sobel_bordas(show_images=True):
    """
    Cria uma imagem sintética com formas (retângulo, círculo), aplica Sobel e
    retorna a magnitude do gradiente. Exibe as imagens.
    """
    # criar imagem em escala de cinza
    H, W = 300, 400
    img = np.zeros((H, W), dtype=np.uint8)
    # retângulo (simula um painel)
    cv2.rectangle(img, (50, 80), (200, 220), color=200, thickness=-1)
    # circulo (simula um objeto)
    cv2.circle(img, (300, 160), 60, color=150, thickness=-1)
    # uma linha fina (simula fio)
    cv2.line(img, (10, 10), (390, 290), color=255, thickness=2)

    # aplica blur leve (ruído realista)
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Sobel nas duas direções
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.convertScaleAbs(magnitude)

    if show_images:
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.imshow(img, cmap='gray'); plt.title('Imagem Sintética'); plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(sobelx, cmap='gray'); plt.title('Sobel X'); plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(magnitude, cmap='gray'); plt.title('Magnitude do Gradiente'); plt.axis('off')
        plt.suptitle('Problema 4 — Detecção de Bordas (Sobel)')
        plt.show()

    return img, magnitude

# ---------------------------
# PROBLEMA 5 — Lote Econômico de Compra (EOQ/LEC) aplicado a hardware TI
# C(q) = (D*S)/q + (H*q)/2  -> derivada C'(q) = - (D*S)/q^2 + H/2 ; resolvendo C'=0
# q* = sqrt((2*D*S) / H)   [nota que a forma clássica é q* = sqrt(2DS / H)]
# ---------------------------
def problema5_eoq(D=10000, S=50.0, H=10.0):
    """
    Calcula o EOQ (lote ótimo) e plota custo total vs q, marcando o mínimo.
    D: demanda anual (unidades)
    S: custo por pedido (fixo)
    H: custo de manter uma unidade por ano
    """
    q = sp.symbols('q', positive=True)
    C = (D * S) / q + (H * q) / 2
    C_prime = sp.diff(C, q)
    q_opt = float(sp.solve(C_prime, q)[0])
    C_opt = float(C.subs(q, q_opt))

    print("Problema 5 — EOQ aplicado ao estoque de hardware")
    print(f"Parâmetros: D={D}, S={S}, H={H}")
    print(f"Lote ótimo q* = {q_opt:.2f} unidades -> Custo anual mínimo = {C_opt:.2f}")

    # plot
    q_vals = np.linspace(max(1, q_opt * 0.2), q_opt * 3, 400)
    C_func = sp.lambdify(q, C, 'numpy')
    plt.figure(figsize=(8,4))
    plt.plot(q_vals, C_func(q_vals), label='Custo Total C(q)')
    plt.scatter([q_opt], [C_opt], c='red', label=f'q*={q_opt:.2f}')
    plt.title('Problema 5 — Lote Econômico de Compra (EOQ)')
    plt.xlabel('Quantidade por pedido (q)')
    plt.ylabel('Custo anual total C(q)')
    plt.legend(); plt.grid(True)
    plt.show()

    # Sensibilidade: variação em S e H
    qs = {}
    for factor in [0.5, 1.0, 2.0]:
        qf = sqrt((2 * D * (S * factor)) / H)
        qs[factor] = qf
    print("Sensibilidade de q* a variações de S (custo por pedido):")
    for f, val in qs.items():
        print(f"  S*{f:.1f} -> q* = {val:.1f}")

    return q_opt, C_opt

# ---------------------------
# MAIN -> executa todos com parâmetros realistas
# ---------------------------
def main():
    print("Executando todos os problemas (cenários reais de TI)\n")
    # Problema 1
    problema1_otimiza_servidor(A=1200.0, B=1.8, C=0.0)

    # Problema 2
    problema2_gradient_descent(learning_rate=0.005, iters=4000)

    # Problema 3
    problema3_adocao_estimativa(K=15000, r=1.0, t0=6.0, noise_scale=400.0)

    # Problema 4
    problema4_sobel_bordas(show_images=True)

    # Problema 5
    problema5_eoq(D=8000, S=75.0, H=12.0)

    print("\nExecução finalizada.")

if __name__ == "__main__":
    main()
