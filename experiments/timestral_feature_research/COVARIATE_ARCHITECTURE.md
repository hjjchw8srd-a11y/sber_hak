# Архитектура Native Covariate Conditioning в Timestral-209M

**Модель:** `mistralai/Timestral-209M`  
**Архитектура:** Patch-based Two-Stream Autoregressive Transformer with Variate Attention  
**Механизм адаптации:** Low-Rank Adaptation (LoRA: $r=8, \alpha=16$, модули `wQKV`, `wO`)  

---

## 1. Концепция многомерного кодирования

В классических авторегрессионных трансформерах временных рядов (таких как Chronos) модель работает исключительно с одномерным целевым рядом $y_t$.  
В `Timestral-209M` реализована нативная поддержка многомерных рядов и экзогенных ковариат через **двухпоточную матричную упаковку каналов (Packed Channel Rows)**.

Каждая временная серия со своими признаками представляется в виде набора параллельных рядов (каналов):
$$R_i = \{ \mathbf{x}_{target}, \mathbf{x}_{past}^{(1)}, \dots, \mathbf{x}_{past}^{(C_p)}, \mathbf{x}_{known}^{(1)}, \dots, \mathbf{x}_{known}^{(C_k)} \}$$

```mermaid
flowchart TD
    subgraph Inputs["Входные каналы серии"]
        T["Целевой ряд (quantity)\n[52 недели]"] -->|ROLE_TARGET = 0| P["Patch Embedding\n(patch_size = 32)"]
        PC["Past Covariates (A, B, C, D, E, G, H)\n[52 недели]"] -->|ROLE_PAST_COV = 1| P
        KC["Known Covariates (Календарь F)\n[61 неделя: 52 + 9]"] -->|ROLE_KNOWN_COV = 2| P
    end

    subgraph Transformer["Трансформер Timestral-209M (24 слоя)"]
        P --> Enc["Encoder Stream (История)"]
        Enc --> TimeAttn["Time Attention\n(Вдоль временной оси патчей)"]
        TimeAttn --> VarAttn["Variate Attention\n(Межканальное внимание между признаками)"]
        VarAttn --> Dec["Decoder Stream (Горизонт 9 недель)"]
    end

    subgraph Masking["Маскирование внимания (No-Leakage)"]
        Mask["Two-Stream Time & Variate Mask\n1. Target future замаскирован\n2. Past Covariates future замаскирован\n3. Known Covariates future доступен как Key/Value"] -.-> TimeAttn
        Mask -.-> VarAttn
    end

    subgraph Output["Выходной слой"]
        Dec --> Head["QuantileHead (21 квантиль)"]
        Head --> TargetOut["Target Quantiles (9 недель)\n[B, 9, 21]"]
        Head --> Loss["Pinball Loss (Loss по target-каналу)"]
    end
```

---

## 2. Роли каналов и маскирование внимания (Attention Roles)

Каждому ряду присваивается роль `role_id`:
1. `ROLE_TARGET (0)`: целевой ряд продаж. На горизонте прогноза декодируются токены запроса прогноза.
2. `ROLE_PAST_COV (1)`: исторические признаки (динамика продаж, остатки, выручка). На горизонте декодера эти каналы маскируются (`key_valid = False`), так как будущие значения неизвестны.
3. `ROLE_KNOWN_COV (2)`: календарные признаки и подтверждённые праздники. Они известны наперёд на все 9 недель горизонта. Их токены в декодере открыты для внимания, поэтому целевые токены прогноза могут напрямую считывать календарный контекст будущих недель!

---

## 3. Дифференцируемый цикл обучения (Differentiable LoRA Training)

Стандартный метод библиотеки `forecaster.forecast()` выполняется в режиме `torch.inference_mode()`.  
Для обеспечения сквозного обучения с градиентами разработан дифференцируемый конвейер через вызов `model.predict_block()`:

1. **Сборка каналов:** функция `_assemble_rows()` упаковывает целевые и ковариатные каналы, дополняя контекст 52 недели слева нулями до кратности `patch_size = 32` (длина $T_{pad} = 64$ патча).
2. **Патчевое масштабирование:** целевой ряд масштабируется каузально per-patch через `causal_patch_loc_scale()`. Известные ковариаты нормализуются через `_known_cov_scaling()`.
3. **Прямой проход:** `predict_block(inputs, observed, group_ids, horizon_patches=1, role_ids, known_future, known_future_mask)`.
4. **Селекция target-канала:** из выходов `(B * V, 32, 21)` извлекаются **исключительно** первые строки каждого батча (`:n_tgt = 1`) и первые 9 временных шагов горизонта (`:9`). Ковариатные строки отбрасываются и не получают потерь.
5. **Целевая функция:** `masked_pinball_loss_21_quantiles` по 21 соревновательному квантилю:
   $$\mathcal{L} = \frac{1}{B \cdot H \cdot Q} \sum_{b, h, q} \max \Big( q (y - \hat{y}_q), (q - 1)(y - \hat{y}_q) \Big)$$
6. **Градиенты LoRA:** оптимизируются адаптерные матрицы ранга 8 в слоях проекций внимания `wQKV` и `wO`.
