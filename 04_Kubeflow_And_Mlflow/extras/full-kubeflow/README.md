# Optional: full Kubeflow (larger instances)

The main module 4 path uses **Kubeflow Pipelines standalone**, which fits a g3.medium. This folder is the
upgrade to the **full Kubeflow platform**: central dashboard with login, per-user namespaces, Notebooks, Katib (hyperparameter
tuning), and more. Needs a **g3.large or bigger and a 100 GB+ disk** (the images alone are ~25 GB; Kubeflow recommends 16 GB RAM / 8 cores for itself).

Do steps 1, 2 and 4 of the main README, then, in place of step 3:

```bash
cd extras/full-kubeflow
./install_full_kubeflow.sh     # 15-30 min
../../port_forward.sh stop     # the standalone UI isn't there any more
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &     # Kubeflow on http://localhost:8080
```

Then continue with step 5. Differences to know: the login is `user@example.com` / `12341234`; pipelines run in the namespace
`kubeflow-user-example-com`, so the MinIO upload and `run_pipeline.py` need that namespace and a logged-in session (not covered here);
the Pipelines UI is inside the dashboard. Pipeline pods still reach module 3's MLflow at the same address. Untested on a cluster:
see `../../INSTRUCTOR_NOTES.md`.
