import json
import numpy as np

# Cargar los datos
def load_data_semantic():
    with open("corpora/cf/json/qrels.json", "r", encoding="utf-8") as f:
        ref_docs = json.load(f)
    with open("resultsCfs.json", "r", encoding="utf-8") as f:
        semantic_docs = json.load(f)

    # El archivo de referencia (qrels) ya debería estar en formato:
    # [{"relevantDocs": [id1, id2, ...]}, ...]

    # Extraer los índices de los artículos recomendados (top 10) por similitud
    semantic_docs_format = [
        {"relevant_docs": [int(doc["relevantDoc"]) for doc in query["relevantDocs"]]}
        for query in semantic_docs
    ]

    return ref_docs, semantic_docs_format
# Métricas
def precision_recall(relDocs, compDocs):
    precision, recall = [], []
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
            prec = hits / (i + 1)
            rec = hits / len(relDocs)
            precision.append(prec)
            recall.append(rec)
    return norm_prec(precision, recall)

def norm_prec(precision, recall):
    newPrecision = [0] * 11
    for i in range(len(recall)):
        for j in range(10):
            if j/10.0 < recall[i] <= (j+1)/10.0:
                newPrecision[j] = max(newPrecision[j], precision[i])
                break
        if recall[i] == 1.0:
            newPrecision[10] = max(newPrecision[10], precision[i])
    p = np.array(newPrecision)
    m = np.zeros(p.size, dtype=bool)
    precision11, excptIndx = [], []
    for i in range(len(newPrecision)):
        m[excptIndx] = True
        a = np.ma.array(p, mask=m)
        precision11.append(newPrecision[np.argmax(a)])
        excptIndx.append(i)
    return precision11

def map_vec(relDocs, compDocs):
    precision = []
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
            prec = hits / (i + 1)
            precision.append(prec)
    return precision

def average_curve(vector):
    ret_vector = [0] * 11
    for j in range(11):
        for i in range(len(vector)):
            ret_vector[j] += vector[i][j]
        ret_vector[j] = float("{:.3f}".format(ret_vector[j] / len(vector)))
    return ret_vector

def p_at_n(relDocs, compDocs, n):
    hits = 0
    if len(compDocs) < n:
        n = len(compDocs)
        if n == 0:
            return 0
    for i in range(n):
        if compDocs[i] in relDocs:
            hits += 1
    return hits / n

def MAP(vPrecision, refDocs):
    totalAP = 0
    for i in range(len(vPrecision)):
        queryAP = 0
        if vPrecision[i]:
            for k in range(len(vPrecision[i])):
                queryAP += vPrecision[i][k]
            queryAP /= len(refDocs[i]["relevantDocs"])
        totalAP += queryAP
    return totalAP / len(vPrecision)

def rprec(relDocs, compDocs):
    hits = 0
    n = len(relDocs)
    if len(compDocs) < n:
        n = len(compDocs)
        if n == 0:
            return 0
    for i in range(n):
        if compDocs[i] in relDocs:
            hits += 1
    return hits / len(relDocs)

def avg_prec_rec(relDocs, compDocs):
    if not relDocs or not compDocs:
        return 0, 0
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits / len(compDocs)
    rec = hits / len(relDocs)
    return prec, rec

def f_beta(avgPrec, avgRec, beta):
    if avgPrec == 0 or avgRec == 0:
        return 0
    return (beta ** 2 + 1) * (avgPrec * avgRec) / (beta ** 2 * avgPrec + avgRec)

def f1(avgPrec, avgRec):
    if avgPrec == 0 or avgRec == 0:
        return 0
    return 2 * ((avgPrec * avgRec) / (avgPrec + avgRec))

# MAIN
if __name__ == "__main__":
    ref_docs, semantic_docs = load_data_semantic()

    map_vals, prec_vals = [], []
    p5, p10, rprec_total, f1_total, fbeta_total = 0, 0, 0, 0, 0
    avg_prec_total, avg_rec_total = 0, 0

    for i in range(len(ref_docs)):
        rel_docs = ref_docs[i]["relevantDocs"]
        sem_docs = semantic_docs[i]["relevant_docs"]

        prec_vals.append(precision_recall(rel_docs, sem_docs))
        map_vals.append(map_vec(rel_docs, sem_docs))
        p5 += p_at_n(rel_docs, sem_docs, 5)
        p10 += p_at_n(rel_docs, sem_docs, 10)

        avg_prec, avg_rec = avg_prec_rec(rel_docs, sem_docs)
        avg_prec_total += avg_prec
        avg_rec_total += avg_rec
        f1_total += f1(avg_prec, avg_rec)
        fbeta_total += f_beta(avg_prec, avg_rec, 2)

        rprec_total += rprec(rel_docs, sem_docs)

    n = len(ref_docs)
    avg_prec_total /= n
    avg_rec_total /= n
    f1_total /= n
    fbeta_total /= n
    p5 /= n
    p10 /= n
    rprec_total /= n
    map_score = MAP(map_vals, ref_docs)

    print(f"📊 Precisión media: {avg_prec_total:.4f}")
    print(f"📊 Recall medio: {avg_rec_total:.4f}")
    print(f"📊 F1: {f1_total:.4f}")
    print(f"📊 F2 (Fbeta): {fbeta_total:.4f}")
    print(f"📊 MAP: {map_score:.4f}")
    print(f"📊 P@5: {p5:.4f}")
    print(f"📊 P@10: {p10:.4f}")
    print(f"📊 R-Precision: {rprec_total:.4f}")