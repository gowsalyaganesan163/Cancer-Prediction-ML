import streamlit as st, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
st.set_page_config(page_title="Breast Cancer Prediction",page_icon="🩺",layout="wide")
st.title("🩺 Breast Cancer Prediction — Logistic Regression")
st.warning("Educational demo only. This model is not a medical diagnosis and must not be used to make clinical decisions.")
st.caption("Upload the breast-cancer dataset. The expected target column is diagnosis (B=Benign, M=Malignant).")
uploaded=st.file_uploader("Upload cancer dataset CSV",type=["csv"])
if uploaded:
    df=pd.read_csv(uploaded)
    st.subheader("Data preview")
    st.dataframe(df.head(10),use_container_width=True)
    if "diagnosis" not in df:
        st.error("Missing required diagnosis column.")
    else:
        st.write(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        st.bar_chart(df["diagnosis"].value_counts())
        if st.button("Train & evaluate Logistic Regression",type="primary"):
            work=df.drop(columns=[c for c in ["id","Unnamed: 32"] if c in df.columns]).copy()
            y=work["diagnosis"].map({"M":1,"B":0})
            X=work.drop(columns="diagnosis").apply(pd.to_numeric,errors="coerce")
            valid=y.notna()
            X=X.loc[valid];y=y.loc[valid]
            X=X.replace([np.inf,-np.inf],np.nan)
            X=X.fillna(X.median())
            if y.nunique()<2: st.error("Both diagnosis classes are required.")
            else:
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
                scaler=StandardScaler()
                Xtr_s=scaler.fit_transform(Xtr);Xte_s=scaler.transform(Xte)
                model=LogisticRegression(max_iter=1000)
                model.fit(Xtr_s,ytr)
                pred=model.predict(Xte_s)
                metrics={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred),"Recall":recall_score(yte,pred),"F1 Score":f1_score(yte,pred)}
                st.subheader("Evaluation")
                cols=st.columns(4)
                for col,(k,v) in zip(cols,metrics.items()): col.metric(k,f"{v:.3f}")
                st.write("Confusion matrix (rows actual B/M, columns predicted B/M)")
                st.dataframe(pd.DataFrame(confusion_matrix(yte,pred),index=["Actual Benign","Actual Malignant"],columns=["Predicted Benign","Predicted Malignant"]))
                st.text(classification_report(yte,pred,target_names=["Benign","Malignant"]))
                results=pd.DataFrame({"Actual":yte.map({0:"Benign",1:"Malignant"}).values,"Predicted":pd.Series(pred,index=yte.index).map({0:"Benign",1:"Malignant"}).values})
                results["Result"]=np.where(yte.values==pred,"Correct","Wrong")
                st.dataframe(results,use_container_width=True)
                st.download_button("Download predictions",results.to_csv(index=False),"cancer_prediction_results.csv","text/csv")
else:
    st.info("Upload your dataset CSV. The source project references a dataset outside the project ZIP, so the app prompts for it.")
