"""Binary Logistic Regression na GPU (torch LBFGS) — drop-in zamiennik sklearn LogisticRegression.

Używane w notatnikach 03 i 04 (warianty `*_torch.ipynb`) do probingu konceptu w aktywacjach CLIP.
Funkcja celu: (1/(2*C*n)) * ||w||^2 + mean_BCE(logits, y) — odpowiednik sklearn.
Intercept nie jest regularyzowany (zgodnie z sklearn domyślnie).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLR(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=200, tol=1e-5, device='cuda', random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = self.device if torch.cuda.is_available() else 'cpu'
        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
        n = Xt.shape[0]
        W = torch.zeros(Xt.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        reg = 1.0 / (2.0 * self.C * n)

        opt = torch.optim.LBFGS(
            [W, b],
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            line_search_fn='strong_wolfe',
        )

        def closure():
            opt.zero_grad()
            logits = Xt @ W + b
            loss = F.binary_cross_entropy_with_logits(logits, yt, reduction='mean') \
                 + reg * (W * W).sum()
            loss.backward()
            return loss

        opt.step(closure)

        self._W = W.detach()
        self._b = b.detach()
        self._device = device
        self.coef_ = self._W.cpu().numpy().reshape(1, -1)
        self.intercept_ = self._b.cpu().numpy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            p = torch.sigmoid(Xt @ self._W + self._b).cpu().numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
