# Errors reference

All public exceptions inherit from `TLDecPyError` and carry a `message` and
optional `hint` string. Call `.to_detail()` to get a serialisable
`ErrorDetail` payload.

```python
try:
    tl.fit_multi(T, I, peaks=[], bg=None)
except tl.TLDecPyError as exc:
    detail = exc.to_detail()
    print(detail.error_code, detail.hint)
```

::: tldecpy.errors.TLDecPyError

---

::: tldecpy.errors.PygcdError

---

::: tldecpy.errors.ModelKeyError

---

::: tldecpy.errors.DomainError

---

::: tldecpy.errors.ConvergenceError

---

::: tldecpy.errors.TypingError

---

::: tldecpy.errors.DatasetError
