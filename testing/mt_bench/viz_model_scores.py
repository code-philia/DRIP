import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]

def get_model_df(score_json_path):
    cnt = 0
    q2result = []
    fin = open(score_json_path, "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df


if __name__ == '__main__':

    target_models = [
                     "Llama-3.2-3B-SpclSpclSpcl-instfuse",
                     "Llama-3.2-3B_SpclSpclSpcl-struq",
                     "Llama-3.2-3B-SpclSpclSpcl-ise",
                     "Llama-3.2-3B_SpclSpclSpcl-secalign"]

    df = pd.DataFrame()
    for model in target_models:
        df2 = get_model_df(f"meta-llama/{model}/gpt-4_judgement_on_mtbench.jsonl")
        df = pd.concat([df, df2])

    all_models = df["model"].unique()
    print(all_models)
    scores_all = []
    for model in all_models:
        res = df[(df["model"] == model) & (df["score"] >= 0)]
        score = res["score"].mean()
        scores_all.append({"model": model, "score": score})
        # for cat in CATEGORIES:
        #     res = df[(df["category"] == cat) & (df["model"] == model) & (df["score"] >= 0)]
        #     score = res["score"].mean()
        #     scores_all.append({"model": model, "category": cat, "score": score})


    scores_target = [scores_all[i] for i in range(len(scores_all)) if (not scores_all[i]["model"].startswith("Llama-3.2-1B")) ]
    print(scores_target)

    # sort by target_models
    scores_target = sorted(scores_target, key=lambda x: x["model"], reverse=True)

    # df_score = pd.DataFrame(scores_target)

    # fig = px.line_polar(df_score, r='score', theta='category', line_close=True,
    #                     category_orders={"category": CATEGORIES},
    #                     color='model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)
    # fig.update_layout(
    #     margin=dict(l=40, r=100, t=40, b=40),  # Increase right margin for legend
    #     legend=dict(
    #         title="",  # Optional: remove legend title
    #
    #     ),
    #     polar=dict(
    #         radialaxis=dict(
    #             showticklabels=True,
    #             ticks='outside',
    #             showline=True
    #         ),
    #         angularaxis=dict(
    #             direction='clockwise'
    #         )
    #     )
    # )
    #
    # fig.write_image("debug.png")