import pandas as pd
import re
import os
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def segment_documents(text, source_label, min_words=30, max_words=150):
    """
    Split full statutory text into paragraph-level chunks for BERTopic.
    Legal paragraphs are bounded by section numbers or double newlines.
    """
    chunks = re.split(r'\n{2,}|\b(?=\d+\.\s+[A-Z])', text)
    
    segments, labels = [], []
    for chunk in chunks:
        words = chunk.split()
        if len(words) < min_words:
            continue
        for start in range(0, len(words), max_words - 20):
            window = ' '.join(words[start:start + max_words])
            if len(window.split()) >= min_words:
                segments.append(window)
                labels.append(source_label)
    return segments, labels

def topic_coherence_pmi(topic_keywords, all_segments, top_n=8):
    """
    Compute average pairwise PMI for top_n keywords of a topic.
    Score > 0 = keywords co-occur more than chance.
    """
    keywords = topic_keywords[:top_n]
    vec = CountVectorizer(vocabulary=keywords, binary=True)
    try:
        X = vec.fit_transform(all_segments).toarray()
    except ValueError:
        return 0.0
    N = len(all_segments)
    
    pmis = []
    for i in range(len(keywords)):
        for j in range(i+1, len(keywords)):
            p_i  = X[:, i].sum() / N + 1e-9
            p_j  = X[:, j].sum() / N + 1e-9
            p_ij = (X[:, i] * X[:, j]).sum() / N + 1e-9
            pmis.append(np.log(p_ij / (p_i * p_j)))
    
    return float(np.mean(pmis)) if pmis else 0.0

class InnovationDiscoverer:
    def __init__(self, ai_dominance_threshold=0.85):
        umap_model  = UMAP(n_neighbors=15, n_components=5,
                           min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5,
                                 metric='euclidean', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words='english', min_df=2)
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=False,
            verbose=True,
            nr_topics='auto'
        )
        self.ai_dominance_threshold = ai_dominance_threshold
        
    def discover_novel_topics(self, documents, document_sources):
        """
        documents: List of text strings (both Human and AI).
        document_sources: List of labels (e.g., 'Primary', 'AI_Draft') corresponding to documents.
        """
        print("Fitting BERTopic on all documents...")
        topics, _ = self.topic_model.fit_transform(documents)
        
        topic_df = pd.DataFrame({
            'Document': documents,
            'Source': document_sources,
            'Topic': topics
        })
        
        novel_topics = []
        
        for topic_id in topic_df['Topic'].unique():
            if topic_id == -1:
                continue
            
            cluster = topic_df[topic_df['Topic'] == topic_id]
            total   = len(cluster)
            ai_count = (cluster['Source'] == 'AI_Draft').sum()
            ai_ratio = ai_count / total
            
            if ai_ratio >= self.ai_dominance_threshold:
                topic_words_raw = self.topic_model.get_topic(topic_id)
                topic_words = [word for word, score in topic_words_raw]
                ai_texts_in_topic = cluster[cluster['Source'] == 'AI_Draft']['Document'].tolist()
                
                novel_topics.append({
                    'topic_id': topic_id,
                    'keywords': topic_words,
                    'ai_ratio': round(ai_ratio, 3),
                    'total_segs': total,
                    'ai_texts': ai_texts_in_topic
                })
                
        return sorted(novel_topics, key=lambda x: x['ai_ratio'], reverse=True)

class GroundingChecker:
    def __init__(self, model_name='nlpaueb/legal-bert-base-uncased', borrow_threshold=0.75):
        print(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.borrow_threshold = borrow_threshold
        self.broader_corpus_embeddings = None
        
    def ingest_broader_corpus(self, broader_texts):
        """
        Embed the broader statutory Indian corpus
        """
        print("Ingesting and embedding broader corpus...")
        self.broader_corpus_embeddings = self.model.encode(broader_texts)
        
    def classify_innovation(self, novel_ai_texts):
        """
        Classifies novel text as 'Regulatory Borrowing' or 'True Hallucination'
        """
        if self.broader_corpus_embeddings is None:
            raise ValueError("Broader corpus not ingested.")
            
        print("Checking grounding of novel AI texts...")
        ai_embeddings = self.model.encode(novel_ai_texts)
        sim_matrix = cosine_similarity(ai_embeddings, self.broader_corpus_embeddings)
        
        results = []
        for i, text in enumerate(novel_ai_texts):
            max_sim = np.max(sim_matrix[i])
            
            if max_sim >= self.borrow_threshold:
                classification = "Regulatory Borrowing"
            else:
                classification = "True Hallucination"
                
            results.append({
                'text_snippet': text[:200] + "...", # Store snippet for review
                'max_grounding_similarity': round(float(max_sim), 4),
                'classification': classification
            })
            
        return results

class CitationAuditor:
    def __init__(self, valid_indian_acts):
        """
        valid_indian_acts: A list of strings containing real Indian Acts 
        """
        self.valid_acts = [act.lower() for act in valid_indian_acts]
        
        self.citation_pattern = re.compile(
            r'\b([A-Z][a-zA-Z\s\(\)]{5,50}'       # Name: starts capital, 5-50 chars
            r'(?:Act|Bill|Rules|Code|Guidelines|Regulations|Ordinance)'
            r'(?:,\s*(?:19|20)\d{2})\b)',          # Year is MANDATORY for citation
            re.IGNORECASE
        )
        
        self.institution_pattern = re.compile(
            r'\b(Ministry of [A-Z][a-zA-Z\s]+|'
            r'National [A-Z][a-zA-Z\s]+ (?:Authority|Board|Commission|Council)|'
            r'Central [A-Z][a-zA-Z\s]+ (?:Authority|Board|Committee))\b'
        )
        
    def citation_is_valid(self, clean_cit, valid_acts, min_overlap=0.6):
        cit_tokens = set(re.findall(r'\w+', clean_cit.lower())) - {'act', 'the', 'of'}
        for valid in valid_acts:
            val_tokens = set(re.findall(r'\w+', valid.lower())) - {'act', 'the', 'of'}
            if not val_tokens:
                continue
            overlap = len(cit_tokens & val_tokens) / len(val_tokens)
            if overlap >= min_overlap:
                return True
        return False
        
    def is_self_referential(self, citation):
        """
        Flag citations to non-existent future or AI-fabricated acts.
        AI drafts often name the very act they are generating as if it exists.
        Any act with year 2024 or later is flagged since these post-date
        the AI training cutoff and are almost certainly fabricated titles.
        """
        year_match = re.search(r'(19|20)(\d{2})', citation)
        if year_match:
            year = int(year_match.group())
            if year >= 2024:
                return True
        return False

    def audit_draft(self, ai_text):
        extracted_citations = self.citation_pattern.findall(ai_text)

        # Deduplicate case-insensitively to avoid double-counting
        # titles that appear in ALL CAPS headings and normal case in body
        seen = set()
        unique_citations = []
        for c in extracted_citations:
            key = c.strip().lower()
            if key not in seen:
                seen.add(key)
                unique_citations.append(c.strip())
        extracted_citations = unique_citations

        if not extracted_citations:
            return {
                'extracted_count': 0,
                'valid_count': 0,
                'self_referential_count': 0,
                'hallucinated_count': 0,
                'self_referential_citations': [],
                'hallucinated_citations': [],
                'hallucination_rate': 0.0,
                'self_referential_rate': 0.0,
            }

        valid_citations = []
        self_referential_citations = []
        hallucinated_citations = []

        for citation in extracted_citations:
            clean_cit = citation.strip().lower()
            if self.is_self_referential(clean_cit):
                self_referential_citations.append(citation.strip())
            elif not self.citation_is_valid(clean_cit, self.valid_acts):
                hallucinated_citations.append(citation.strip())
            else:
                valid_citations.append(citation.strip())

        total = len(extracted_citations)

        return {
            'extracted_count': total,
            'valid_count': len(valid_citations),
            'self_referential_count': len(self_referential_citations),
            'hallucinated_count': len(hallucinated_citations),
            'self_referential_citations': self_referential_citations,
            'hallucinated_citations': hallucinated_citations,
            'hallucination_rate': round(len(hallucinated_citations) / total, 2),
            'self_referential_rate': round(len(self_referential_citations) / total, 2),
        }

if __name__ == "__main__":
    
    VALID_INDIAN_ACTS = [
        # Core baselines
        "Digital Personal Data Protection Act, 2023",
        "Air Prevention and Control of Pollution Act, 1981",
        "National Security Act, 1980",

        # Data Protection adjacents
        "Information Technology Act, 2000",
        "Information Technology Amendment Act, 2008",
        "Aadhaar Act, 2016",
        "Right to Information Act, 2005",
        "Telecom Regulatory Authority of India Act, 1997",
        "Credit Information Companies Regulation Act, 2005",
        "Payment and Settlement Systems Act, 2007",
        "Personal Data Protection Bill, 2019",
        "DPDP Draft Rules, 2025",

        # Air Pollution adjacents
        "Environment Protection Act, 1986",
        "Water Prevention and Control of Pollution Act, 1974",
        "National Green Tribunal Act, 2010",
        "Motor Vehicles Act, 1988",
        "Central Motor Vehicles Rules, 1989",
        "Energy Conservation Act, 2001",
        "Noise Pollution Regulation and Control Rules, 2000",
        "Public Liability Insurance Act, 1991",
        "Forest Conservation Act, 1980",
        "National Capital Region Planning Board Act, 1985",
        "Environment Impact Assessment Notification, 2006",
        "Industrial Emission Standards, 2012",
        "National Ambient Air Quality Standards, 2009",

        # National Security adjacents
        "Unlawful Activities Prevention Act, 1967",
        "Armed Forces Special Powers Act, 1958",
        "Prevention of Money Laundering Act, 2002",
        "Prevention of Corruption Act, 1988",
        "Bharatiya Nyaya Sanhita, 2023",
        "National Cyber Security Policy, 2013",
        "Indian Penal Code, 1860",
        "Code of Criminal Procedure, 1973",

        # Cross-domain common references
        "Constitution of India",
        "Indian Contract Act, 1872",
        "General Clauses Act, 1897",
        "Companies Act, 2013",

        # Shorthand references AI commonly uses
        "Air Act, 1981",
        "Foreigners Act, 1946",
        "NCRPB Act, 1985",
        "IT Rules, 2011",
        "Information Technology Rules, 2011",
        "Information Technology Intermediary Guidelines Rules, 2021",
        "Sensitive Personal Data or Information Rules, 2011",
        "National Green Tribunal Act, 2010",
        "Water Act, 1974",
        "Noise Pollution Rules, 2000",
    ]

    # Load master dataframe
    input_csv = 'result/final_outputs/04_master_data.csv' 
    print(f"Loading dataframe from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
         print(f"Error: {input_csv} not found. Ensure previous stages are run.")
         exit(1)

    documents = []
    sources = []
    all_segments = []
    all_sources = []

    for index, row in df.iterrows():
        file_path = row['file']
        source_type = 'AI_Draft' if row['source'] == 'ai_draft' else 'Primary'
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(text)
                sources.append(source_type)
                
                segs, labs = segment_documents(text, source_type)
                all_segments.extend(segs)
                all_sources.extend(labs)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    # Add core baselines manually if they aren't marked correctly in the DataFrame
    baseline_paths = [
        r"extracted_text\core_baselines\Digital Personal Data Protection Act, 2023.txt",
        r"extracted_text\core_baselines\Air (Prevention and Control of Pollution) Act, 1981.txt",
        r"extracted_text\core_baselines\National Security Act, 1980.txt"
    ]
    for path in baseline_paths:
         try:
            with open(path, 'r', encoding='utf-8') as f:
                 text = f.read()
                 documents.append(text)
                 sources.append("Primary")
                 
                 segs, labs = segment_documents(text, "Primary")
                 all_segments.extend(segs)
                 all_sources.extend(labs)
         except Exception as e:
            pass

    print(f"Total segments for BERTopic: {len(all_segments)}")

    print("\n--- 1. Running BERTopic Innovation Discovery ---")
    discoverer = InnovationDiscoverer()
    novel_topics = discoverer.discover_novel_topics(all_segments, all_sources)
    print(f"Found {len(novel_topics)} AI-invented topics.")
    
    coherent_novel = [
        t for t in novel_topics
        if topic_coherence_pmi(t['keywords'], all_segments) > 0
    ]
    print(f"Coherent novel topics (PMI > 0): {len(coherent_novel)} of {len(novel_topics)}")

    print("\n--- 2. Running Grounding Check via Sentence Transformers ---")
    checker = GroundingChecker()

    BROADER_CORPUS_PATHS = {
        "Data Protection": [
            r"extracted_text\broader_corpus\DPDP\Information Technology Act, 2000.txt",
            r"extracted_text\broader_corpus\DPDP\IT amendment act 2008.txt",
            r"extracted_text\broader_corpus\DPDP\Aadhar.txt",
            r"extracted_text\broader_corpus\DPDP\Right to information act.txt",
            r"extracted_text\broader_corpus\DPDP\Telecom Regulatory Authority of India Act, 1997.txt",
            r"extracted_text\broader_corpus\DPDP\Credit Information Companies Regulation Act, 2005.txt",
            r"extracted_text\broader_corpus\DPDP\Payment and Settlement Systems Act, 2007.txt",
            r"extracted_text\broader_corpus\DPDP\Personal Data Protection Bill, 2019.txt",
            r"extracted_text\broader_corpus\DPDP\DPDP Draft Rules.txt",
            r"extracted_text\broader_corpus\DPDP\Justice B. N. Srikrishna Committee Report on Data Protection (2018).txt",
            r"extracted_text\broader_corpus\DPDP\NITI Aayog reports on data governance.txt",
            r"extracted_text\broader_corpus\DPDP\Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021.txt",
        ],
        "Air Pollution": [
            r"extracted_text\broader_corpus\Air pollution\Environment (Protection) Act, 1986.txt",
            r"extracted_text\broader_corpus\Air pollution\Water (Prevention and Control of Pollution) Act, 1974.txt",
            r"extracted_text\broader_corpus\Air pollution\National Green Tribunal Act, 2010.txt",
            r"extracted_text\broader_corpus\Air pollution\Motor Vehicles Act, 1988 (emissions standards).txt",
            r"extracted_text\broader_corpus\Air pollution\Energy Conservation Act, 2001.txt",
            r"extracted_text\broader_corpus\Air pollution\Noise Pollution (Regulation and Control) Rules, 2000.txt",
            r"extracted_text\broader_corpus\Air pollution\Public Liability Insurance Act, 1991.txt",
            r"extracted_text\broader_corpus\Air pollution\forest conservation act.txt",
            r"extracted_text\broader_corpus\Air pollution\ep_act_1986.txt",
            r"extracted_text\broader_corpus\Air pollution\Industrial Emission Standards.txt",
            r"extracted_text\broader_corpus\Air pollution\National_Ambient_Air_Quality_Standards.txt",
            r"extracted_text\broader_corpus\Air pollution\dpcc report o air pollution.txt",
            r"extracted_text\broader_corpus\Air pollution\Air-Pollution-report by dpcc.txt",
            r"extracted_text\broader_corpus\Air pollution\Comprehensive-Action-Plan on delhi air pollution.txt",
            r"extracted_text\broader_corpus\Air pollution\steps taken to abate air pollution in delhi.txt",
            r"extracted_text\broader_corpus\Air pollution\Supreme Court orders on Delhi pollution.txt",
            r"extracted_text\broader_corpus\Air pollution\Grap.txt",
            r"extracted_text\broader_corpus\Air pollution\NGT (National Green Tribunal) rulings on AQI.txt",
            r"extracted_text\broader_corpus\Air pollution\Order-graded-response-action-plan grap.txt",
        ],
        "National Security": [
            r"extracted_text\broader_corpus\National security\Unlawful Activities (Prevention) Act, 1967.txt",
            r"extracted_text\broader_corpus\National security\Armed Forces Special Powers Act, 1958.txt",
            r"extracted_text\broader_corpus\National security\Prevention of Money Laundering Act, 2002.txt",
            r"extracted_text\broader_corpus\National security\prevention of corruption 1988.txt",
            r"extracted_text\broader_corpus\National security\Bharatiya Nyaya Sanhita, 2023.txt",
            r"extracted_text\broader_corpus\National security\National_cyber_security_policy-2013_0.txt",
            r"extracted_text\broader_corpus\National security\terrorism.txt",
        ]
    }

    broader_corpus_texts = []
    missing_files = []

    for domain, paths in BROADER_CORPUS_PATHS.items():
        domain_loaded = 0
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content.split()) < 50:
                        print(f"[warn] File too short, skipping: {path}")
                        continue
                    broader_corpus_texts.append(content)
                    domain_loaded += 1
            except FileNotFoundError:
                missing_files.append(path)
            except Exception as e:
                print(f"[warn] Could not read {path}: {e}")
        print(f"  [{domain}] Loaded {domain_loaded} of {len(paths)} files")

    if missing_files:
        print(f"\n[warn] {len(missing_files)} files not found:")
        for f in missing_files:
            print(f"  MISSING: {f}")

    if len(broader_corpus_texts) == 0:
        raise RuntimeError(
            "No broader corpus documents loaded. "
            "Check file paths in BROADER_CORPUS_PATHS."
        )

    print(f"\nBroader corpus: {len(broader_corpus_texts)} documents loaded, "
          f"{sum(len(t.split()) for t in broader_corpus_texts):,} total words")

    checker.ingest_broader_corpus(broader_corpus_texts)

    novel_reports = []
    for topic_dict in novel_topics:
         print(f"\nEvaluating Topic {topic_dict['topic_id']} (Keywords: {', '.join(topic_dict['keywords'][:5])})...")
         sample_texts = topic_dict['ai_texts'][:5]
         classifications = checker.classify_innovation(sample_texts)
         
         for c in classifications:
              print(f"  Classification: {c['classification']} (Sim: {c['max_grounding_similarity']}) - {c['text_snippet'][:50]}...")
              novel_reports.append(c)

    print("\n--- 3. Running Anachronism & Citation Audit ---")
    auditor = CitationAuditor(VALID_INDIAN_ACTS)

    # Audit the full DataFrame of AI Drafts specifically
    hallucination_rates = []
    self_referential_rates = []
    drafts_with_citations = []
    
    all_extracted_count = 0
    all_hallucinated = []
    all_self_ref = []

    for d, s in zip(documents, sources):
        if s == 'AI_Draft':
            audit_result = auditor.audit_draft(d)
            hallucination_rates.append(audit_result['hallucination_rate'])
            self_referential_rates.append(audit_result['self_referential_rate'])
            if audit_result['extracted_count'] > 0:
                drafts_with_citations.append(audit_result)
            
            all_hallucinated.extend(audit_result.get('hallucinated_citations', []))
            all_self_ref.extend(audit_result.get('self_referential_citations', []))
            all_extracted_count += audit_result['extracted_count']

    ai_draft_count = len([s for s in sources if s == 'AI_Draft']) or 1

    avg_hallucination_overall    = sum(hallucination_rates) / ai_draft_count
    avg_self_referential_overall = sum(self_referential_rates) / ai_draft_count

    cond_hallucination = (
        sum(r['hallucination_rate'] for r in drafts_with_citations) /
        len(drafts_with_citations)
    ) if drafts_with_citations else 0.0

    cond_self_referential = (
        sum(r['self_referential_rate'] for r in drafts_with_citations) /
        len(drafts_with_citations)
    ) if drafts_with_citations else 0.0

    print(f"\n=== CITATION AUDIT RESULTS ===")
    print(f"AI drafts audited:               {ai_draft_count}")
    print(f"Drafts containing citations:     {len(drafts_with_citations)}")
    print(f"\nOverall rates (all AI drafts):")
    print(f"  True Hallucination Rate:       {avg_hallucination_overall*100:.1f}%")
    print(f"  Self-Referential Rate:         {avg_self_referential_overall*100:.1f}%")
    print(f"\nConditional rates (citing drafts only):")
    print(f"  True Hallucination Rate:       {cond_hallucination*100:.1f}%")
    print(f"  Self-Referential Rate:         {cond_self_referential*100:.1f}%")

    # Diagnostics — sample output for verification
    print("\n--- CITATION AUDIT DIAGNOSTICS ---")

    print(f"\nTotal unique citations extracted: {all_extracted_count}")
    print(f"Self-referential (AI naming own act): {len(all_self_ref)}")
    print(f"True hallucinations (fabricated real acts): {len(all_hallucinated)}")

    print("\nSample TRUE HALLUCINATIONS (first 10):")
    for c in all_hallucinated[:10]:
        print(f"  '{c}'")

    print("\nSample SELF-REFERENTIAL citations (first 10):")
    for c in all_self_ref[:10]:
        print(f"  '{c}'")

    print("\nStage 6 Dimension 5 Code execution complete.")
