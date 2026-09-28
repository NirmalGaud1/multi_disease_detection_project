import os
import sys
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import gdown


st.set_page_config(
    page_title="MDRS-Net++ Multi-Disease Risk Stratification",
    layout="wide"
)


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ePN-Dvt5p54aEAVdkU5qsEEbcNtVnnhI?usp=sharing"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mdrs_net_final_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "mdrs_net_metadata.json")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def stratify_risk(probs):
    tiers_def = [
        ("Low", 0.00, 0.25),
        ("Moderate", 0.25, 0.50),
        ("High", 0.50, 0.75),
        ("Critical", 0.75, 1.01),
    ]
    tiers = np.empty(len(probs), dtype=object)
    for name, lo, hi in tiers_def:
        mask = (probs >= lo) & (probs < hi)
        tiers[mask] = name
    return tiers


RISK_TIERS = [
    ("Low", 0.00, 0.25),
    ("Moderate", 0.25, 0.50),
    ("High", 0.50, 0.75),
    ("Critical", 0.75, 1.01),
]


class Linear:
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.T @ dy
        self.db = dy.sum(axis=0)
        return dy @ self.W.T


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return relu(x)

    def backward(self, dy):
        return dy * relu_grad(self.x)


class Dropout:
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training and self.p > 0:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        self.mask = np.ones_like(x)
        return x

    def backward(self, dy):
        return dy * self.mask


class DFIG:
    def __init__(self, in_dim):
        self.w = np.ones(in_dim) * 0.5
        self.rscale = np.array(0.3)
        self.dw = np.zeros_like(self.w)
        self.drscale = np.array(0.0)
        self.cache = None

    def forward(self, x, enabled=True):
        if not enabled:
            self.cache = ("bypass",)
            return x
        signx = np.sign(x)
        absx = np.abs(x)
        a = absx * self.w
        g = sigmoid(a)
        gated = g * x
        resid = self.rscale * self.w * x
        y = gated + resid
        self.cache = ("active", x, g, absx, signx)
        return y

    def backward(self, dy):
        mode = self.cache[0]
        if mode == "bypass":
            return dy
        _, x, g, absx, signx = self.cache
        dg_da = g * (1 - g)
        da_dx = self.w * signx
        dgated_dx = g + x * dg_da * da_dx
        dresid_dx = self.rscale * self.w
        dx = dy * (dgated_dx + dresid_dx)
        dw_total = dy * (x * dg_da * absx + self.rscale * x)
        self.dw = dw_total.sum(axis=0)
        drscale = dy * (self.w * x)
        self.drscale = drscale.sum()
        return dx

    def feature_attribution(self, x, enabled=True):
        if not enabled:
            return np.zeros_like(x)
        absx = np.abs(x)
        g = sigmoid(absx * self.w)
        gated = g * x
        resid = self.rscale * self.w * x
        contrib = gated + resid
        denom = np.sum(np.abs(contrib), axis=1, keepdims=True) + 1e-9
        return contrib / denom


class TRE:
    def __init__(self, hidden_dim=24):
        self.hidden_dim = hidden_dim
        h2 = hidden_dim // 2
        freqs = np.arange(h2).astype(np.float64)
        self.denom = 10000.0 ** (2.0 * freqs / hidden_dim)
        self.linear = Linear(hidden_dim, hidden_dim)
        self.cache = None

    def forward(self, age, enabled=True):
        B = age.shape[0]
        if not enabled:
            self.cache = ("bypass",)
            return np.zeros((B, self.hidden_dim))
        age_norm = np.clip(age, 1, 120) / 120.0
        angles = age_norm[:, None] / self.denom[None, :]
        decay = np.exp(-0.5 * (1 - age_norm))[:, None]
        sin_enc = np.sin(angles) * decay
        cos_enc = np.cos(angles) * decay
        emb_raw = np.concatenate([sin_enc, cos_enc], axis=-1)
        out = self.linear.forward(emb_raw)
        self.cache = ("active",)
        return out

    def backward(self, dy):
        if self.cache[0] == "bypass":
            return
        self.linear.backward(dy)


class CrossAttention:
    def __init__(self, dim):
        self.dim = dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.scale = np.sqrt(dim)
        self.cache = None

    def forward(self, mod_a, mod_b, enabled=True):
        B = mod_a.shape[0]
        if not enabled:
            self.cache = ("bypass",)
            return np.concatenate([mod_a, mod_b], axis=-1)
        X = np.stack([mod_a, mod_b], axis=1)
        Xflat = X.reshape(B * 2, self.dim)
        Qf = self.q.forward(Xflat)
        Kf = self.k.forward(Xflat)
        Vf = self.v.forward(Xflat)
        Q = Qf.reshape(B, 2, self.dim)
        K = Kf.reshape(B, 2, self.dim)
        V = Vf.reshape(B, 2, self.dim)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / self.scale
        scores_shift = scores - scores.max(axis=-1, keepdims=True)
        expx = np.exp(scores_shift)
        A = expx / expx.sum(axis=-1, keepdims=True)
        O = np.matmul(A, V)
        out = O.reshape(B, 2 * self.dim)
        self.cache = ("active", B, Q, K, V, A)
        return out

    def backward(self, dOut):
        if self.cache[0] == "bypass":
            d = dOut.shape[-1] // 2
            return dOut[:, :d], dOut[:, d:]
        _, B, Q, K, V, A = self.cache
        dO = dOut.reshape(B, 2, self.dim)
        dA = np.matmul(dO, V.transpose(0, 2, 1))
        dV = np.matmul(A.transpose(0, 2, 1), dO)
        s = (dA * A).sum(axis=-1, keepdims=True)
        dscores = A * (dA - s)
        dscores = dscores / self.scale
        dQ = np.matmul(dscores, K)
        dK = np.matmul(dscores.transpose(0, 2, 1), Q)
        dQf = dQ.reshape(B * 2, self.dim)
        dKf = dK.reshape(B * 2, self.dim)
        dVf = dV.reshape(B * 2, self.dim)
        dXf_q = self.q.backward(dQf)
        dXf_k = self.k.backward(dKf)
        dXf_v = self.v.backward(dVf)
        dXf = dXf_q + dXf_k + dXf_v
        dX = dXf.reshape(B, 2, self.dim)
        return dX[:, 0, :], dX[:, 1, :]

    def attention_weights(self):
        if self.cache[0] == "bypass":
            return None
        return self.cache[-1]


class DiseaseBranch:
    def __init__(self, raw_dim, common_dim=64, tre_dim=24, age_idx=None):
        self.age_idx = age_idx
        self.dfig = DFIG(raw_dim)
        self.tre = TRE(hidden_dim=tre_dim)
        self.mod_a_proj = Linear(raw_dim, common_dim)
        self.mod_b_proj = Linear(tre_dim, common_dim)
        self.fusion = CrossAttention(common_dim)
        self.out_dim = common_dim * 2

    def forward(self, x, use_dfig=True, use_tre=True, use_fusion=True):
        gated = self.dfig.forward(x, enabled=use_dfig)
        if self.age_idx is not None:
            age_col = x[:, self.age_idx]
        else:
            age_col = np.full(x.shape[0], 50.0)
        tre_emb = self.tre.forward(age_col, enabled=use_tre)
        mod_a = self.mod_a_proj.forward(gated)
        mod_b = self.mod_b_proj.forward(tre_emb)
        fused = self.fusion.forward(mod_a, mod_b, enabled=use_fusion)
        return fused

    def backward(self, dfused, use_dfig=True, use_tre=True, use_fusion=True):
        dmod_a, dmod_b = self.fusion.backward(dfused)
        dgated = self.mod_a_proj.backward(dmod_a)
        dtre = self.mod_b_proj.backward(dmod_b)
        self.tre.backward(dtre)
        self.dfig.backward(dgated)


class Trunk:
    def __init__(self, in_dim, hidden=128):
        self.l1 = Linear(in_dim, hidden)
        self.r1 = ReLU()
        self.d1 = Dropout(0.25)
        self.l2 = Linear(hidden, hidden // 2)
        self.r2 = ReLU()
        self.d2 = Dropout(0.15)
        self.out_dim = hidden // 2

    def forward(self, x, training=True):
        h = self.l1.forward(x)
        h = self.r1.forward(h)
        h = self.d1.forward(h, training)
        h = self.l2.forward(h)
        h = self.r2.forward(h)
        h = self.d2.forward(h, training)
        return h

    def backward(self, dh):
        dh = self.d2.backward(dh)
        dh = self.r2.backward(dh)
        dh = self.l2.backward(dh)
        dh = self.d1.backward(dh)
        dh = self.r1.backward(dh)
        dh = self.l1.backward(dh)
        return dh


class IndividualHead:
    def __init__(self, in_dim, hidden=64):
        self.l1 = Linear(in_dim, hidden)
        self.r1 = ReLU()
        self.d1 = Dropout(0.2)
        self.l2 = Linear(hidden, 1)

    def forward(self, x, training=True):
        h = self.l1.forward(x)
        h = self.r1.forward(h)
        h = self.d1.forward(h, training)
        h = self.l2.forward(h)
        return h[:, 0]

    def backward(self, dlogit):
        dh = dlogit[:, None]
        dh = self.l2.backward(dh)
        dh = self.d1.backward(dh)
        dh = self.r1.backward(dh)
        dh = self.l1.backward(dh)
        return dh


class MDRSNetNumpy:
    def __init__(self, disease_dims, age_indices=None, common_dim=64,
                 trunk_hidden=128, tre_dim=24, use_shared_trunk=True):
        self.use_shared_trunk = use_shared_trunk
        self.disease_dims = disease_dims
        self.common_dim = common_dim
        self.trunk_hidden = trunk_hidden
        self.tre_dim = tre_dim
        age_indices = age_indices or {}
        self.age_indices = age_indices
        self.branches = {
            k: DiseaseBranch(d, common_dim, tre_dim, age_indices.get(k))
            for k, d in disease_dims.items()
        }
        branch_out = common_dim * 2
        if use_shared_trunk:
            self.trunk = Trunk(branch_out, trunk_hidden)
            self.heads = {k: Linear(self.trunk.out_dim, 1) for k in disease_dims}
        else:
            self.heads = {k: IndividualHead(branch_out) for k in disease_dims}

    def forward(self, x, key, use_dfig=True, use_tre=True, use_fusion=True, training=True):
        fused = self.branches[key].forward(x, use_dfig, use_tre, use_fusion)
        if self.use_shared_trunk:
            rep = self.trunk.forward(fused, training)
            logit = self.heads[key].forward(rep)[:, 0]
        else:
            logit = self.heads[key].forward(fused, training)
        return logit

    def backward(self, dlogit, key, use_dfig=True, use_tre=True, use_fusion=True):
        if self.use_shared_trunk:
            dh = self.heads[key].backward(dlogit[:, None])
            dfused = self.trunk.backward(dh)
        else:
            dfused = self.heads[key].backward(dlogit)
        self.branches[key].backward(dfused, use_dfig, use_tre, use_fusion)

    def explain(self, x, key, use_dfig=True):
        return self.branches[key].dfig.feature_attribution(x, enabled=use_dfig)


sys.modules["mdrs_net_lib"] = sys.modules[__name__]


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        return

    try:
        gdown.download_folder(
            DRIVE_FOLDER_URL,
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )
    except Exception as e:
        st.error(
            "Unable to download model files from Google Drive.\n\n"
            "Make sure the Google Drive folder is shared as "
            "'Anyone with the link - Viewer'."
        )
        st.exception(e)
        st.stop()

    model_found = None
    meta_found = None

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            if file == "mdrs_net_final_model.pkl":
                model_found = full_path
            elif file == "mdrs_net_metadata.json":
                meta_found = full_path

    if model_found and model_found != MODEL_PATH:
        shutil.copy2(model_found, MODEL_PATH)

    if meta_found and meta_found != META_PATH:
        shutil.copy2(meta_found, META_PATH)


download_models()

if not os.path.exists(MODEL_PATH):
    st.error("mdrs_net_final_model.pkl was not found in the Google Drive folder.")
    st.stop()

if not os.path.exists(META_PATH):
    st.error("mdrs_net_metadata.json was not found in the Google Drive folder.")
    st.stop()


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return bundle, meta


try:
    bundle, meta = load_bundle()
except Exception as e:
    st.error("Unable to load the MDRS-Net++ model.")
    st.exception(e)
    st.stop()


model = bundle["model"]
scalers = bundle["scalers"]
feats = bundle["feats"]
encoders = bundle["encoders"]
imputers = bundle["imputer"]
labels = bundle["labels"]


st.title("MDRS-Net++ Multi-Disease Risk Stratification")

st.markdown(
    """
    **Multi-Disease AI Risk Prediction and Stratification System**

    Select a disease, enter the patient information, and obtain the
    predicted risk probability, risk tier, and the most influential
    features contributing to the prediction.
    """
)

disease_keys = list(labels.keys())
disease_display = [labels[k] for k in disease_keys]

selected_display = st.sidebar.selectbox("Select Disease", disease_display)
selected_key = disease_keys[disease_display.index(selected_display)]

st.sidebar.markdown("---")
st.sidebar.write(f"**Number of features:** {len(feats[selected_key])}")
st.sidebar.markdown("---")
st.sidebar.info("Model: MDRS-Net++")

st.subheader(f"Patient Input — {labels[selected_key]}")

feature_list = feats[selected_key]
encoder_dict = encoders[selected_key]
imputer = imputers[selected_key]

col1, col2 = st.columns(2)
input_values = {}

for i, feat in enumerate(feature_list):
    target_col = col1 if i % 2 == 0 else col2

    if feat in encoder_dict:
        le = encoder_dict[feat]
        options = list(le.classes_)
        val = target_col.selectbox(feat, options, key=f"{selected_key}_{feat}")
        input_values[feat] = le.transform([val])[0]
    else:
        try:
            idx = feature_list.index(feat)
            default_val = float(imputer.statistics_[idx])
        except Exception:
            default_val = 0.0
        val = target_col.number_input(feat, value=default_val, key=f"{selected_key}_{feat}")
        input_values[feat] = val

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    try:
        row = pd.DataFrame([input_values])[feature_list]
        X_imputed = imputer.transform(row)
        X_scaled = scalers[selected_key].transform(X_imputed)
        logits = model.forward(X_scaled, selected_key, training=False)
        prob = float(sigmoid(logits)[0])
        tier = stratify_risk(np.array([prob]))[0]
        attributions = model.explain(X_scaled, selected_key)
        order = np.argsort(-np.abs(attributions[0]))[:5]
        top_feats = [(feature_list[j], float(attributions[0, j])) for j in order]

        st.markdown("---")
        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Risk Probability", f"{prob * 100:.2f}%")

        with c2:
            st.metric("Risk Tier", tier)

        with c3:
            if prob < 0.25:
                status = "Low Risk"
            elif prob < 0.50:
                status = "Moderate Risk"
            elif prob < 0.75:
                status = "High Risk"
            else:
                status = "Critical Risk"
            st.metric("Risk Status", status)

        st.markdown("---")
        st.subheader("Top Contributing Features")

        contrib_df = pd.DataFrame(top_feats, columns=["Feature", "Attribution"])
        chart_df = contrib_df.set_index("Feature")

        st.bar_chart(chart_df)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Interpretation")

        if tier == "Low":
            st.success(
                "The model indicates a relatively low predicted risk "
                "for the selected disease."
            )
        elif tier == "Moderate":
            st.warning(
                "The model indicates a moderate predicted risk. "
                "Further assessment may be appropriate."
            )
        elif tier == "High":
            st.warning(
                "The model indicates a high predicted risk. "
                "Additional clinical evaluation is recommended."
            )
        else:
            st.error(
                "The model indicates a critical predicted risk. "
                "Professional clinical assessment should be considered."
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.markdown("---")
st.subheader("Model Performance Summary")

final_metrics = bundle.get("final_metrics", {})

if selected_key in final_metrics:
    m = final_metrics[selected_key]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("AUC", f"{m.get('AUC', float('nan')):.3f}")
    mc2.metric("F1", f"{m.get('F1', float('nan')):.3f}")
    mc3.metric("Precision", f"{m.get('Precision', float('nan')):.3f}")
    mc4.metric("Recall", f"{m.get('Recall', float('nan')):.3f}")
else:
    st.info("Performance metrics are not available for the selected disease.")

st.markdown("---")
st.caption("MDRS-Net++ Multi-Disease Risk Stratification System")
