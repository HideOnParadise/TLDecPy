# Kinetic model functions

Model functions are vectorised callables: `f(T, Im, E, Tm, ...) -> ndarray`.
Obtain any model at runtime with `tl.get_model("fo_rq")`.

## First-order (FO)

::: tldecpy.models.fo.fo_rq

---

::: tldecpy.models.fo.fo_rb

---

::: tldecpy.models.fo.fo_ka

---

::: tldecpy.models.fo.fo_wp

## Second-order (SO)

::: tldecpy.models.so.so_ks

---

::: tldecpy.models.so.so_la

## General-order (GO)

::: tldecpy.models.go.go_kg

---

::: tldecpy.models.go.go_rq

---

::: tldecpy.models.go.go_ge

## Mixed-order (MO)

::: tldecpy.models.mixed.mo_kitis

---

::: tldecpy.models.mixed.mo_quad

---

::: tldecpy.models.mixed.mo_vej

## OTOR

::: tldecpy.models.otor_lw.otor_lw

---

::: tldecpy.models.otor_wo.otor_wo

## Continuous trap distributions

::: tldecpy.models.continuous.continuous_gaussian

---

::: tldecpy.models.continuous.continuous_exponential

## Registry utilities

::: tldecpy.models.registry.get_model

---

::: tldecpy.models.registry.list_models

---

::: tldecpy.models.registry.get_model_info

---

::: tldecpy.models.registry.ModelInfo
