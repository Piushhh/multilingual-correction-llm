"""
Language Detection Evaluation (Task 12 / Milestone 6).

Evaluates the hybrid script-composition + statistical language detector
on a benchmark of >=150 labeled examples across English, Hindi, and code-mixed text.
Performs threshold ablation and generates paper/tables/language_detection_results.md
and paper/figures/language_detection_confusion_matrix.png.
"""

import json
from pathlib import Path
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.document_ai.evaluation.metrics import evaluate_language_detection
from src.document_ai.language.detector import detect_language

TABLES_DIR = Path("paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Labeled evaluation dataset (>= 50 per class -> >= 150 total)
LABELED_CORPUS = [
    # English (52 samples)
    ("The architecture of modern deep neural networks relies on high-dimensional vector representations.", "en"),
    ("Gradient descent with momentum accelerates convergence along flat optimization valleys.", "en"),
    ("Self-attention matrices compute scaled dot-product similarities across key and query embeddings.", "en"),
    ("Backpropagation computes partial derivatives of the loss function using recursive chain rule applications.", "en"),
    ("A convolutional filter slides over input feature maps to capture localized spatial hierarchies.", "en"),
    ("Operating systems provide virtual memory abstraction through page tables and hardware address translation.", "en"),
    ("Relational databases enforce ACID guarantees to prevent concurrency anomalies during transactions.", "en"),
    ("Binary search trees maintain logarithmic depth when balanced using tree rotation invariants.", "en"),
    ("Asymptotic computational complexity quantifies algorithm runtime growth as inputs approach infinity.", "en"),
    ("A compiler translates high-level programming syntax into executable low-level machine instructions.", "en"),
    ("Eigenvalues represent scalar multipliers associated with invariant eigenvectors under linear transformation.", "en"),
    ("Definite integrals evaluate the accumulation of continuous quantities across bounded geometric intervals.", "en"),
    ("The central limit theorem establishes that properly normalized sums of random variables converge to Gaussian distributions.", "en"),
    ("Metric spaces define rigorous distance functions satisfying the fundamental triangle inequality axiom.", "en"),
    ("Differential manifolds provide mathematical frameworks for generalized multivariable calculus on smooth curved surfaces.", "en"),
    ("Regularization methods penalize parameter magnitudes to mitigate severe machine learning overfitting.", "en"),
    ("Recurrent neural networks propagate internal hidden states across sequential temporal timestamps.", "en"),
    ("Generative adversarial networks pit generative submodels against discriminative adversaries in minimax games.", "en"),
    ("Distributed consensus protocols achieve fault-tolerant state agreement across unreliable networks.", "en"),
    ("Public key cryptography pairs private decryptors with public encryptors for secure internet exchanges.", "en"),
    ("Data normalization scales raw numeric feature ranges to standardize variance and speed convergence.", "en"),
    ("Batch normalization computes mini-batch means and standard deviations to smooth the optimization landscape.", "en"),
    ("Vision transformers partition raw images into sequential visual patches for self-attention blocks.", "en"),
    ("Quantization reduces floating point memory precision down to compact integer arithmetic.", "en"),
    ("Parameter efficient fine-tuning modifies minimal low-rank matrices instead of whole foundational weights.", "en"),
    ("Hash collisions occur when distinct input keys map onto the identical storage index in memory.", "en"),
    ("Quick sort partitions arrays recursively around chosen pivot elements with expected linearithmic time.", "en"),
    ("Semaphores protect concurrent threads from entering overlapping critical code sections simultaneously.", "en"),
    ("Dynamic programming breaks complex optimization problems down into recursive overlapping subproblems.", "en"),
    ("Turing machines model computational decidability and theoretical algorithmic capabilities.", "en"),
    ("The determinant indicates whether square matrix columns form linearly independent bases.", "en"),
    ("Singular value decomposition factors real matrices into orthogonal and diagonal matrix factors.", "en"),
    ("Cauchy sequences converge in complete metric spaces under standard topological distance norms.", "en"),
    ("Partial differential equations model physical transport phenomena including heat diffusion and fluid dynamics.", "en"),
    ("Bayesian inference updates prior belief distributions upon conditioning on observed empirical evidence.", "en"),
    ("Software testing frameworks ensure automated regression detection across continuous delivery pipelines.", "en"),
    ("Microservices communicate over lightweight network protocols like REST and asynchronous message brokers.", "en"),
    ("Containerization encapsulates runtime application dependencies into isolated immutable execution layers.", "en"),
    ("Load balancers distribute user network requests across pools of redundant backend service hosts.", "en"),
    ("Deadlock prevention algorithms eliminate circular wait conditions from system resource graphs.", "en"),
    ("The university library offers quiet study rooms, reference books, and research journal archives.", "en"),
    ("Public transportation systems reduce urban vehicle emissions and highway traffic congestion.", "en"),
    ("Daily cardiovascular exercise improves metabolic health and mental cognitive performance.", "en"),
    ("Weather forecasts predict sunshine and pleasant spring temperatures over the coming weekend.", "en"),
    ("Local farmers sell organic fruits, heirloom vegetables, and artisan cheese at the neighborhood market.", "en"),
    ("The art gallery exhibits contemporary sculptures, abstract paintings, and historic photographs.", "en"),
    ("Solar energy systems harness photovoltaic solar panels to generate sustainable residential electricity.", "en"),
    ("Volunteers gathered to plant indigenous shade trees along the dry eroded riverbank.", "en"),
    ("The symphony orchestra rehearsed classic orchestral masterpieces for the upcoming seasonal concert.", "en"),
    ("Proper hydration and restful sleep hygiene are essential foundations for immune health.", "en"),
    ("Historic preservation societies protect architectural landmarks from commercial urban redevelopment.", "en"),
    ("Community youth clubs encourage sportsmanship, artistic creativity, and civic engagement.", "en"),

    # Hindi (52 samples)
    ("गहन शिक्षण में न्यूरल नेटवर्क डेटा की जटिल विशेषताओं को स्वतः निकालने में सक्षम होते हैं।", "hi"),
    ("ग्रेडिएंट डिसेंट एल्गोरिदम हानि फलन के न्यूनतम मान तक पहुंचने के लिए भारों को अद्यतित करता है।", "hi"),
    ("ट्रांसफार्मर आर्किटेक्चर ध्यान तंत्र का उपयोग करके शब्दों के बीच संबंधों का विश्लेषण करता है।", "hi"),
    ("बैकप्रोपेगेशन प्रत्येक परत के लिए आंशिक अवकलजों की गणना करने की एक कुशल विधि है।", "hi"),
    ("कन्वोल्यूशनल न्यूरल नेटवर्क छवियों में दृश्य पैटर्न और किनारों को पहचानने के लिए आदर्श हैं।", "hi"),
    ("कंप्यूटर विज्ञान में डेटा संरचनाएं सूचना को कुशलतापूर्वक संग्रहीत करने का आधार हैं।", "hi"),
    ("बाइनरी सर्च सॉर्ट की गई सूची में तत्वों को खोजने के लिए लॉगरिदमिक समय जटिलता लेता है।", "hi"),
    ("ऑपरेटिंग सिस्टम कंप्यूटर की हार्डवेयर मेमोरी और प्रक्रियाओं को प्रबंधित करता है।", "hi"),
    ("रिलेशनल डेटाबेस तालिकाओं में डेटा संग्रहीत करते हैं और प्रश्नों का त्वरित उत्तर देते हैं।", "hi"),
    ("डेडलॉक तब होता है जब दो प्रक्रियाएं एक-दूसरे द्वारा रोके गए संसाधनों की प्रतीक्षा करती हैं।", "hi"),
    ("रैखिक बीजगणित सदिश अंतरिक्ष और आव्यूह परिवर्तनों का गहन अध्ययन करता है।", "hi"),
    ("अवकलन किसी चर के सापेक्ष फलन के परिवर्तन की तात्कालिक दर को व्यक्त करता है।", "hi"),
    ("समाकलन वक्र के नीचे के कुल संचित क्षेत्रफल की गणना करने में उपयोगी होता है।", "hi"),
    ("आइगेनवेल्यू और आइगेनवेक्टर रैखिक प्रणालियों के मूलभूत अभिलक्षण होते हैं।", "hi"),
    ("प्रायिकता सिद्धांत अनिश्चित और यादृच्छिक घटनाओं के गणितीय विश्लेषण की नींव है।", "hi"),
    ("अति-अनुकूलन को कम करने के लिए मॉडल में नियमितीकरण और ड्रॉपआउट लागू किया जाता है।", "hi"),
    ("सॉफ्टमैक्स फलन आउटपुट मानों को एक वैध संभाव्यता वितरण में परिवर्तित करता है।", "hi"),
    ("ऑटोएनकोडर इनपुट का संकुचित प्रतिनिधित्व सीखकर उसे पुनः निर्मित करने का प्रयास करता है।", "hi"),
    ("आवर्ती तंत्रिका नेटवर्क समय श्रृंखला और अनुक्रमिक भाषा डेटा के प्रसंस्करण के लिए बने हैं।", "hi"),
    ("एडम एल्गोरिदम गति और अनुकूली सीखने की दरों को मिलाकर त्वरित अभिसरण सुनिश्चित करता है।", "hi"),
    ("हैश टेबल त्वरित लुकअप और प्रविष्टि के लिए औसत स्थिर समय जटिलता प्रदान करती है।", "hi"),
    ("क्विक सॉर्ट विभाजन विधि का उपयोग करके बड़े एरे को तेजी से क्रमबद्ध करता है।", "hi"),
    ("मल्टीथ्रेडिंग एक ही प्रोग्राम में कई कार्यों को समानांतर में चलाने की सुविधा देती है।", "hi"),
    ("डायनामिक प्रोग्रामिंग जटिल समस्याओं को उप-समस्याओं में तोड़कर कुशलता से हल करती है।", "hi"),
    ("कंपाइलर स्रोत कोड का विश्लेषण करके उसे मशीन निष्पादन योग्य बाइनरी कोड में बदलता है।", "hi"),
    ("सारणिक एक संख्यात्मक मान है जो मैट्रिक्स के व्युत्क्रमणीय होने की जानकारी देता है।", "hi"),
    ("केंद्रीय सीमा प्रमेय बताता है कि बड़े नमूनों का औसत सामान्य वितरण का अनुसरण करता है।", "hi"),
    ("बेयस प्रमेय नई जानकारी मिलने पर पूर्व संभावनाओं को अद्यतन करने का सूत्र है।", "hi"),
    ("टेलर श्रृंखला बहुपदों की मदद से गैर-रैखिक फलनों का स्थानीय सन्निकटन प्रदान करती है।", "hi"),
    ("सदिश अंतरिक्ष में सदिश जोड़ और स्केलर गुणन के नियम पूरी तरह से संतुष्ट होते हैं।", "hi"),
    ("विश्वविद्यालय का पुस्तकालय छात्रों के अध्ययन और शोध के लिए अनेक पुस्तकें उपलब्ध कराता है।", "hi"),
    ("सार्वजनिक परिवहन का उपयोग करने से नगरों में वायु प्रदूषण और भीड़भाड़ में कमी आती है।", "hi"),
    ("प्राकृतिक पर्यावरण की रक्षा करना हम सभी नागरिकों का एक महत्वपूर्ण सामूहिक दायित्व है।", "hi"),
    ("संतुलित भोजन और ताजे फल शरीर को आवश्यक पोषक तत्व और ऊर्जा प्रदान करते हैं।", "hi"),
    ("मौसम विभाग ने कल सुबह घने कोहरे और ठंडी हवाओं की चेतावनी जारी की है।", "hi"),
    ("विद्यालय के छात्रों ने वार्षिक खेल दिवस पर अपनी प्रतिभा का उत्कृष्ट प्रदर्शन किया।", "hi"),
    ("प्राचीन भारतीय वास्तुकला के मंदिर अपनी भव्य नक्काशी और सुंदरता के लिए प्रसिद्ध हैं।", "hi"),
    ("नदियों का स्वच्छ जल कृषि और ग्रामीण आजीविका के लिए अमूल्य प्राकृतिक संसाधन है।", "hi"),
    ("योग और प्राणायाम का दैनिक अभ्यास शारीरिक और मानसिक स्वास्थ्य को सुदृढ़ बनाता है।", "hi"),
    ("सड़क सुरक्षा नियमों का कड़ाई से पालन करने से सड़क दुर्घटनाओं को रोका जा सकता है।", "hi"),
    ("वन्यजीव अभयारण्य दुर्लभ पशुओं और पक्षियों को सुरक्षित प्राकृतिक आवास प्रदान करते हैं।", "hi"),
    ("गांवों में सौर ऊर्जा संयंत्र स्थापित होने से बिजली की समस्या का स्थायी समाधान हुआ है।", "hi"),
    ("हस्तशिल्प और खादी मेले में स्थानीय कारीगरों द्वारा बनाए गए सुंदर उत्पाद प्रदर्शित हुए।", "hi"),
    ("पुस्तकों का नियमित पठन ज्ञान में वृद्धि करता है और एकाग्रता को बढ़ाता है।", "hi"),
    ("स्वयंसेवकों ने नगर निगम के सहयोग से शहर के मुख्य पार्क की सफाई की।", "hi"),
    ("कृषि वैज्ञानिकों ने किसानों को कम पानी में अधिक पैदावार देने वाले बीजों की सलाह दी।", "hi"),
    ("रेलवे स्टेशन पर यात्रियों की सुविधा के लिए स्वचालित टिकट मशीनें लगाई गई हैं।", "hi"),
    ("ऐतिहासिक संग्रहालय में मध्यकालीन सिक्के और दुर्लभ पांडुलिपियां सुरक्षित रखी गई हैं।", "hi"),
    ("सवेरे की सैर और ताजी हवा शरीर में ताजगी और सकारात्मक ऊर्जा का संचार करती है।", "hi"),
    ("समुदाय के बुजुर्गों के अनुभव और मार्गदर्शन से युवा पीढ़ी को प्रेरणा मिलती है।", "hi"),
    ("जल संरक्षण के लिए वर्षा जल संचयन प्रणाली को घरों में अपनाना जरूरी है।", "hi"),
    ("सांस्कृतिक उत्सव में लोकगीतों और पारंपरिक नृत्यों की रंगारंग प्रस्तुति हुई।", "hi"),

    # Code-Mixed (Hindi-English) (52 samples)
    ("यह transformer model multi-head attention mechanism का बहुत effective उपयोग करता है।", "code_mixed"),
    ("Deep learning में backpropagation की मदद से weights और biases optimize होते हैं।", "code_mixed"),
    ("Neural network training के दौरान cross-entropy loss gradients calculate किए जाते हैं।", "code_mixed"),
    ("Overfitting रोकने के लिए dropout layer और L2 regularization का प्रयोग किया जाता है।", "code_mixed"),
    ("Convolutional layers raw image data से high-level features extract करती हैं।", "code_mixed"),
    ("Adam optimizer adaptive learning rate use करके loss function को minimize करता है।", "code_mixed"),
    ("Pretrained language model को downstream classification tasks पर fine-tune किया जाता है।", "code_mixed"),
    ("Softmax layer model के output logits को probability distribution में convert करती है।", "code_mixed"),
    ("Residual connections deep networks में vanishing gradient issue को solve करते हैं।", "code_mixed"),
    ("Vector space में embeddings semantic similarity capture करने में helpful होती हैं।", "code_mixed"),
    ("Binary search algorithm sorted array में target element search करने के लिए O(log n) time लेता है।", "code_mixed"),
    ("Operating system processes के memory allocation और context switching को control करता है।", "code_mixed"),
    ("Relational database systems transactions की ACID properties ensure करते हैं।", "code_mixed"),
    ("Multithreading environment में data race prevent करने के लिए mutex lock लगाते हैं।", "code_mixed"),
    ("Dynamic programming problems में subproblems के answers memoize किए जाते हैं।", "code_mixed"),
    ("Compiler source code parse करके abstract syntax tree construct करता है।", "code_mixed"),
    ("Distributed systems में network partition के time CAP theorem apply होती है।", "code_mixed"),
    ("Microservices REST APIs और message queue के through asynchronously interact करती हैं।", "code_mixed"),
    ("Continuous integration pipeline automated tests run करके pull requests verify करती है।", "code_mixed"),
    ("Kubernetes cluster containers की deployment, scaling और health check manage करता है।", "code_mixed"),
    ("Linear algebra में eigenvalues और eigenvectors linear transformation analyze करने में use होते हैं।", "code_mixed"),
    ("Derivative calculate करके function की instantaneous rate of change measure की जाती है।", "code_mixed"),
    ("Definite integral curve के under total bounded area compute करता है।", "code_mixed"),
    ("Matrix invertible तभी होती है जब उसका determinant non-zero value रखे।", "code_mixed"),
    ("Central limit theorem states कि sample mean normal distribution में converge करता है।", "code_mixed"),
    ("Bayesian probability prior distribution को experimental data से update करती है।", "code_mixed"),
    ("Singular value decomposition (SVD) matrix factorization की important technique है।", "code_mixed"),
    ("Gradient vector multivariable function के steepest ascent की direction show करता है।", "code_mixed"),
    ("Jacobian matrix first-order partial derivatives को systematic table में organize करता है।", "code_mixed"),
    ("Hessian matrix local curvature और optimization convexity check करने के काम आता है।", "code_mixed"),
    ("Diffusion model gradual noise removal के through high resolution image generate करता है।", "code_mixed"),
    ("LoRA technique parameters freeze करके low rank matrices train करती है।", "code_mixed"),
    ("Quantization float32 weights को int8 में convert करके memory requirement कम करता है।", "code_mixed"),
    ("FlashAttention memory bandwidth optimize करके execution time dramatically reduce करता है।", "code_mixed"),
    ("Direct Preference Optimization (DPO) human preferences के अनुसार language model align करता है।", "code_mixed"),
    ("RAG pipeline semantic retrieval use करके LLM hallucinations prevent करती है।", "code_mixed"),
    ("Vector database embeddings index करने के लिए approximate nearest neighbor search use करता है।", "code_mixed"),
    ("Prompt engineering में few-shot examples model generation quality improve करते हैं।", "code_mixed"),
    ("Cosine similarity metric two embedding vectors के बीच angle measure करता है।", "code_mixed"),
    ("Knowledge distillation teacher model की capabilities compact student model में transfer करता है।", "code_mixed"),
    ("University campus में new student computer center और sports ground open हुआ है।", "code_mixed"),
    ("Weekend trip के लिए online train ticket booking app use करना बहुत convenient है।", "code_mixed"),
    ("Public transport use करने से traffic jam और air pollution significantly reduce होता है।", "code_mixed"),
    ("Morning walk और healthy diet mental peace और overall wellness improve करती हैं।", "code_mixed"),
    ("Museum exhibition में ancient artifacts और historical manuscripts showcase किए गए हैं।", "code_mixed"),
    ("Local market में handmade crafts और traditional souvenirs की shopping tourists बहुत like करते हैं।", "code_mixed"),
    ("Mobile application download करके electricity bill payment easily घर बैठे हो जाती है।", "code_mixed"),
    ("Hospital emergency ward में advanced medical devices और trained doctors available हैं।", "code_mixed"),
    ("Solar rooftop panels install करवाने पर government special subsidy provide कर रही है।", "code_mixed"),
    ("Weather forecast ने आने वाले weekend पर clear sky और pleasant temperature predict किया है।", "code_mixed"),
    ("Library membership card renew कराकर online digital books borrow की जा सकती हैं।", "code_mixed"),
    ("Community awareness drive में school children ने environmental protection पर play enact किया।", "code_mixed"),
]


def run_evaluation():
    print(f"Loaded {len(LABELED_CORPUS)} labeled language samples.")
    lang_counts = {}
    for _, lang in LABELED_CORPUS:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print("Class distribution:", lang_counts)

    # Threshold Ablation Study
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ablation_results = {}

    for th in thresholds:
        correct = 0
        per_class_correct = {"en": 0, "hi": 0, "code_mixed": 0}
        per_class_total = {"en": 0, "hi": 0, "code_mixed": 0}

        for text, true_lang in LABELED_CORPUS:
            pred = detect_language(text, code_mixed_threshold=th)["language"]
            per_class_total[true_lang] += 1
            if pred == true_lang:
                correct += 1
                per_class_correct[true_lang] += 1

        acc = correct / len(LABELED_CORPUS)
        ablation_results[th] = {
            "overall_acc": acc,
            "en_acc": per_class_correct["en"] / per_class_total["en"],
            "hi_acc": per_class_correct["hi"] / per_class_total["hi"],
            "cm_acc": per_class_correct["code_mixed"] / per_class_total["code_mixed"],
        }
        print(f"Threshold {th:.2f}: Overall Acc = {acc*100:.2f}% | EN: {ablation_results[th]['en_acc']*100:.1f}%, HI: {ablation_results[th]['hi_acc']*100:.1f}%, CM: {ablation_results[th]['cm_acc']*100:.1f}%")

    # Evaluate at standard production threshold (0.15)
    default_th = 0.15
    classes = ["en", "hi", "code_mixed"]
    conf_matrix = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}

    for text, true_lang in LABELED_CORPUS:
        pred_lang = detect_language(text, code_mixed_threshold=default_th)["language"]
        if pred_lang in conf_matrix[true_lang]:
            conf_matrix[true_lang][pred_lang] += 1
        else:
            # unexpected / unknown
            pass

    # Confusion matrix array
    cm_array = np.array([[conf_matrix[t][p] for p in classes] for t in classes])

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=["English (en)", "Hindi (hi)", "Code-Mixed (cm)"],
        yticklabels=["English (en)", "Hindi (hi)", "Code-Mixed (cm)"],
        title="Language Detector Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    thresh = cm_array.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm_array[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_array[i, j] > thresh else "black")

    fig.tight_layout()
    cm_path = FIGURES_DIR / "language_detection_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Output Markdown Table
    out_table = TABLES_DIR / "language_detection_results.md"
    best_overall = ablation_results[default_th]["overall_acc"]

    with open(out_table, "w", encoding="utf-8") as f:
        f.write("# Multilingual Language Detection Evaluation (Task 12)\n\n")
        f.write(f"- **Benchmark Size:** {len(LABELED_CORPUS)} curated samples (English: {lang_counts['en']}, Hindi: {lang_counts['hi']}, Code-Mixed: {lang_counts['code_mixed']})\n")
        f.write(f"- **Overall Accuracy (threshold=0.15):** {best_overall * 100:.2f}%\n")
        f.write("- **Methodology:** Dual-signal hybrid combining unicode script composition (Devanagari vs Latin ratios) with statistical Latin verification.\n\n")

        f.write("### Confusion Matrix (Production Threshold = 0.15)\n\n")
        f.write("| True \\ Predicted | English (`en`) | Hindi (`hi`) | Code-Mixed (`code_mixed`) | Recall (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for i, c_true in enumerate(classes):
            row_sum = sum(conf_matrix[c_true].values())
            rec = (conf_matrix[c_true][c_true] / row_sum * 100) if row_sum else 0
            f.write(f"| **{c_true}** | {conf_matrix[c_true]['en']} | {conf_matrix[c_true]['hi']} | {conf_matrix[c_true]['code_mixed']} | {rec:.1f}% |\n")

        f.write("\n### Code-Mixed Sensitivity Threshold Ablation\n\n")
        f.write("| Threshold (Minority Ratio) | Overall Accuracy (%) | English Acc (%) | Hindi Acc (%) | Code-Mixed Acc (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for th, metrics in ablation_results.items():
            mark = " **(Production Default)**" if th == 0.15 else ""
            f.write(f"| `{th:.2f}`{mark} | {metrics['overall_acc']*100:.2f}% | {metrics['en_acc']*100:.1f}% | {metrics['hi_acc']*100:.1f}% | {metrics['cm_acc']*100:.1f}% |\n")

    print(f"Saved evaluation table to {out_table}")


if __name__ == "__main__":
    run_evaluation()
