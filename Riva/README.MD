# Cybercrime Complaint Data Classification & Analysis

## Overview
This project aims to develop a comprehensive taxonomy of cybercrime categories and subcategories. By accurately classifying each complaint based on its nature and severity, we bridge the gap between how citizens describe cybercrime incidents and how law enforcement requires structured data for effective action. Ultimately, this leads to more actionable reports and improved response times.

## Objectives
- **Comprehensive Taxonomy:** Develop clear categories and subcategories for cybercrime complaints.
- **Enhanced Data Quality:** Clean and structure complaint data to resolve linguistic inconsistencies and standardize terminology.
- **Actionable Insights:** Enable law enforcement to respond more effectively by transforming unstructured citizen reports into clear, actionable information.
- **Empowered Decision-Making:** Provide data-driven insights for better crime prevention and resolution.

## Challenges
- **Multilingual Data:** Complaints are written in a mix of languages and incorporate multiple linguistic codes, complicating interpretation.
- **Ambiguity & Inconsistency:** Vague descriptions, ambiguous explanations, and inconsistent terminology hinder accurate classification.
- **Data Noise:** Unnecessary characters, punctuation, and irrelevant numeric data add noise that must be removed for effective analysis.

## Dataset Overview
The dataset contains key features essential for analysis and modeling:

- **Complaint ID:**  
  Each complaint is assigned a unique identification number, ensuring accurate tracking and data integrity.

- **Crime Description:**  
  Contains unstructured textual narratives detailing the incident, the involved parties, and contextual circumstances. This feature requires extensive preprocessing (e.g., text normalization, tokenization, vectorization) for effective utilization.

- **Time Stamp:**  
  Captures the exact moment the complaint was registered, including hour, minute, day, month, and year. This facilitates time-based analysis, such as trend and seasonal pattern identification.

## Methodology

### Exploratory Data Analysis (EDA)
- **Purpose:**  
  To understand the underlying structure of the dataset by uncovering patterns, trends, and anomalies.
- **Approach:**  
  - Analyzing the distribution of crime categories and subcategories.
  - Identifying data quality issues and informing subsequent preprocessing decisions.

### Data Preprocessing
To enhance data reliability and prepare it for modeling, several preprocessing steps were undertaken:

#### 1. Handling Missing Values
- **Identification:**  
  Detailed review to pinpoint missing or null entries.
- **Evaluation:**  
  Each missing entry was assessed within its context to determine its impact.
- **Action:**  
  Non-essential gaps were removed, and a secondary review ensured data completeness and accuracy.

#### 2. Noise Removal and Text Cleaning
- **Cleaning Process:**  
  - Removal of extraneous special characters and punctuation.
  - Elimination of redundant whitespace (spaces, tabs, line breaks).
  - Exclusion of irrelevant numeric data.
- **Outcome:**  
  Streamlined text that focuses on pertinent linguistic content, significantly improving the quality of subsequent analyses.

### Feature Engineering & Feature Selection
- **Insight Extraction:**  
  Transforming cleaned textual data into meaningful features.
- **Model Preparation:**  
  Selecting the most relevant features to enhance model reliability and predictive accuracy.

## Impact & Benefits
- **For Citizens:**  
  Simplifies the grievance filing process, making it more intuitive and efficient.
- **For Law Enforcement:**  
  Provides clear, actionable data that empowers better response times and data-driven decision-making in crime prevention and resolution.

## Data Filtering and Language Restriction
Within the scope of the hackathon's problem statement, the dataset was refined to include only entries written in Hindi, English, and Hinglish. This was accomplished by filtering the text using ASCII encoding, which retains only characters in the Roman script. Consequently, languages that use non-Roman scripts—such as Telugu, Tamil, and others—were effectively excluded. This approach proved both effective and computationally simpler compared to methods employing libraries like LangGid or LanDetect, particularly for detecting Indic languages.

## Taxonomy Refinement: Subcategory Clarifications
To enhance clarity and ensure actionable categorization, several subcategories with unclear or vague names were revised. The following table summarizes the original subcategory names alongside their updated versions:


| **Original Name**                                                  | **Revised Name**                                                   |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Rape/Gang Rape                                                     | Content Related to Rape/Gang Rape                                  |
| Sexual Harassment                                                  | Online Sexual Coercion & Harassment                                |
| Fake News                                                          | Fake News & Misinformation                                         |
| Cyber Stalking & Bullying                                          | Cyber Stalking and Cyber Bullying                                  |
| Publishing and Transmitting Obscene Material/Sexually Explicit Material | Publishing & Transmitting Obscene/Sexually Explicit Material   |
| Cheating by Impersonation (other than Government Servant)          | Impersonation Fraud (Non-Government)                               |
| Sextortion/Nude Video Call                                         | Sextortion/Nude Video Call Extortion                               |
| Aadhar Enabled Payment System (AEPS) Fraud/Biometric Cloning         | Aadhar/Biometric Cloning Fraud                                     |
| Business Email Compromise (BEC) Fraud                              | Business Email Compromise/Fraudulent Takeover                      |
| Human Trafficking                                                  | Human Trafficking via Dark Web/Cyber Slavery                       |
| IPR Theft                                                          | Intellectual Property Theft (IPR)                                  |

This taxonomy refinement ensures that each category is clearly defined and accurately reflects the nature of the complaint, facilitating better communication between citizens and law enforcement, and improving the overall actionability of the reports.

## Exploratory Data Analysis (EDA): Statistical and Visualization Techniques

A variety of statistical and visualization methods were applied to the dataset to extract meaningful insights and guide subsequent modeling steps. These techniques helped identify patterns, trends, and anomalies in the crime descriptions, ultimately informing both feature selection and data preprocessing strategies.

### 1. Word Cloud Visualization
- **Objective:**  
  Identify the most frequently used words in the crime descriptions.
- **Outcome:**  
  Common keywords such as "fraud," "account," "money," "bank," "credit," "card," "victim," "UPI," "transaction," and "caller" emerged, suggesting a strong focus on financial fraud and related cybercrime activities.

### 2. Text-Based Statistical Analysis
- **Whitespace Count:**  
  Measured the number of spaces in each text entry to understand variations in sentence structure.
- **Word Count Distribution:**  
  Analyzed the distribution of word counts to detect reporting patterns; most descriptions were found to be short, with a few extended cases that could indicate detailed fraud reports or complex cases.
- **Character Length Distribution:**  
  Examined text length variations to identify potential noise or inconsistencies. The majority of descriptions fell between 100 and 300 characters, with a long tail indicating some significantly longer entries.

### 3. Crime Temporal Analysis
- **Day vs. Night Crime Reported Frequency:**  
  Categorized complaints by time period, revealing that night-time crimes are significantly higher than those reported during the day. This may indicate that certain cyber crimes, such as fraud or scams, tend to occur more frequently at night.
- **Crime Reporting Frequency by Hour of the Day:**  
  Analyzed crime occurrences by hour, showing:
  - **Early Morning (Midnight to 6 AM):** Lowest crime frequency.
  - **Late Morning to Afternoon (Post-8 AM to 6 PM):** A sharp rise in incidents, corresponding with high activity periods.
  - **Evening (After 8 PM):** A gradual decline, although rates remain significant until about 10 PM.
- **Crime Reporting Frequency by Month of the Year:**  
  Revealed seasonal trends with peaks in the last quarter (September to December) and relatively high counts in May and June, suggesting possible influences from external factors such as law enforcement activities, holidays, or weather conditions.

### 4. N-Gram Analysis and Key Insights
- **Bigram & Trigram Analysis (Top 10 Most Frequent Phrases):**  
  - Frequent trigrams such as "asked me to," "please help me," "a call from," and "take necessary action" were identified.
  - These phrases often suggest financial fraud, distress calls, coercion, or scam-related activities.
  - Phrases like "total amount in" and "reverse total amount" were linked to discussions about financial transactions, potentially indicating banking fraud or online scams.
- **Additional Insights:**
  - **Whitespace and Word Count Distributions:**  
    Both were right-skewed, with the majority of entries being concise. A minority of cases showed unusually high whitespace counts or extended word counts, hinting at detailed narratives.
  - **Temporal Patterns:**  
    The analysis confirmed that certain types of cybercrime may have a higher occurrence during night-time, potentially due to reduced vigilance or increased opportunities for social engineering scams.
  - **Textual Complexity in `crimeaditionalinfo`:**  
    Detailed analysis using TF-IDF scoring and n-gram evaluation helped distill the most relevant words and phrases, refining the preprocessing strategy to handle varying reporting styles and levels of detail.

### Conclusion
The comprehensive EDA, combining both statistical measures and visualization techniques, was instrumental in uncovering intricate details of the dataset. By understanding the frequency distributions, temporal patterns, and common linguistic features, the team was able to enhance data preprocessing and feature selection, ultimately leading to more accurate classification and actionable insights for law enforcement.

## Advanced Classification Pipeline

Our classification process is organized into three key phases, each designed to incrementally refine the categorization of crime descriptions:

### 1. Topic Keyword Generation with LDA
- **Objective:**  
  Extract topic-specific keywords from crime descriptions to aid in classification.
- **Methodology:**  
  - Applied Latent Dirichlet Allocation (LDA) to extract keywords.
  - Generated over 200+ keywords per subcategory in English, Hindi, and Hinglish.
  - Manually curated the descriptions for each category and subcategory to ensure clarity.
  - Augmented the keyword lists with examples from the sample labeled data for better representation.

### 2. Embedding and Semantic Matching
- **Objective:**  
  Leverage semantic similarity to accurately match crime descriptions with their relevant categories.
- **Methodology:**  
  - Stored the generated descriptions, keywords, and examples in JSON files, organized by category and subcategory.
  - Created word embeddings for these keywords.
  - Employed cosine similarity to perform semantic matching between the crime descriptions (from the `crimeAdditionalInfo` dataframe) and the stored keywords.
  - Derived a semantic score that determined the most relevant category and subcategory for each crime description.

### 3. Classification with BiLSTM
- **Objective:**  
  Enhance the contextual understanding of crime descriptions for final classification.
- **Methodology:**  
  - Fed the semantically matched data into a Bidirectional LSTM (BiLSTM) model.
  - The BiLSTM captured both past and future dependencies in the text, refining the contextual nuances.
  - This deep learning approach, combined with the previous statistical techniques, ensured robust and accurate categorization into the designated categories and subcategories.

This integrated approach—merging topic modeling with semantic matching and deep learning—ensures our system delivers precise and actionable classifications for cybercrime complaints.

## PIPELINE
Pipeline
├── Preprocessing
│   ├── Before Preprocessing
│   │   ├── Big Reports
│   ├── Preprocessing Steps
│   ├── After Preprocessing
│   │   ├── Exploratory Data Analysis
│
├── Embeddings
│   ├── Extract Embeddings
│   ├── Embedding Model
│   │   ├── embeddings_model.pt
│   │   ├── tokenizer.pkl
│
├── Model
│   ├── Semantic-Op
│   │   ├── Semantic Checker
│   │   │   ├── vector_description
│   ├── Classification
│   │   ├── Comparison of Classified Crime Text
│   │   ├── JSON File
│
├── Crime Classification
│   ├── Pretraining with LSTM Model

## Model Building & Selection Justification

Throughout our experimentation, we explored multiple techniques to optimize the classification of crime descriptions into relevant categories and subcategories. Below are the iterations we tested, the challenges encountered, and the rationale behind our final approach.

### Iterative Approaches and Challenges

1. **Clustering-Enhanced Classification: K-Means, BiLSTM & Autoencoder**
   - **Approach:**  
     - Applied K-Means clustering to group crime descriptions based on similarity.
     - Employed a BiLSTM model for classification.
     - Used an autoencoder for dimensionality reduction.
   - **Challenges:**  
     - K-Means struggled with high-dimensional textual data, resulting in poorly defined clusters.
     - BiLSTM alone did not generalize effectively across subcategories without strong supervision.
     - The autoencoder caused a loss of semantic information, thereby reducing classification accuracy.

2. **Embeddings-Driven Classification with Custom BERT**
   - **Approach:**  
     - Generated embeddings for crime descriptions.
     - Developed a custom BERT model from scratch for classification.
   - **Challenges:**  
     - BERT exhibited limitations in handling domain-specific crime data, leading to misclassifications.

3. **Adaptive Classification with Confidence Scoring & Autoencoder**
   - **Approach:**  
     - Implemented an adaptive classifier that incorporated confidence scoring to improve category assignments.
     - Coupled this with an autoencoder for feature extraction.
   - **Challenges:**  
     - The confidence score thresholding introduced errors, especially for crimes with ambiguous labels.
     - The autoencoder again caused semantic loss, affecting overall reliability.

4. **Structured Categorization with Hierarchical Clustering & Custom BERT**
   - **Approach:**  
     - Used hierarchical clustering on embeddings to establish a structured categorization.
     - Followed this with a custom BERT model for further refinement.
   - **Challenges:**  
     - BERT’s lack of interpretability made it difficult to analyze and rectify incorrect classifications.

### Why Our Methodology is Better

Our refined approach integrates several complementary techniques:
- **Accurate Keyword Generation:**  
  - Leveraged LDA to extract over 200+ topic-specific keywords per subcategory in English, Hindi, and Hinglish.
  - Manually curated category descriptions and augmented them with labeled examples for enhanced clarity.
- **Custom Sentence Embeddings:**  
  - Developed embeddings tailored to our multilingual corpus, improving contextual and keyword mapping.
- **Improved Semantic Matching:**  
  - Employed cosine similarity between crime descriptions and predefined category embeddings for an effective first-pass classification.
- **Context-Aware Classification:**  
  - Used a BiLSTM model trained on our refined dataset to capture both past and future dependencies, ensuring precise classification.
- **Preservation of Semantic Information:**  
  - Unlike autoencoders that resulted in semantic distortion, our approach retains contextual meaning at every stage.
- **Computational Efficiency:**  
  - The two-stage system (semantic matching followed by BiLSTM verification) strikes a balance between efficiency and high classification accuracy.

### Scalability & Real-World Integration for the NCPR Portal

- **Scalability for Large Datasets:**  
  Designed to handle millions of crime records by reducing deep processing via efficient semantic matching.
- **Context Awareness:**  
  Sentence-level embeddings capture multilingual and domain-specific nuances for precise categorization.
- **Security & Compliance:**  
  - No reliance on external pre-trained models or third-party dependencies ensures complete data privacy.
  - Custom-trained components meet NCPR’s legal and security requirements.
- **Domain-Specific Adaptability:**  
  Trained on a crime-specific corpus, our model avoids the misclassification issues common in generic models.
- **Fair Categorization:**  
  Special efforts have been made to accurately classify crimes against vulnerable groups (e.g., women and children) through nuanced embeddings that capture contextual subtleties.

### Model Evaluation Metrics & Parameters

- **Semantic Matching:**  
  Cosine similarity is calculated between text representations and predefined category embeddings to predict both the primary category and subcategory.
- **BiLSTM Validation:**  
  A BiLSTM model validates the semantic matching results, achieving an accuracy of **89.96%**, which reinforces the reliability of our classification approach.

### Model Training and Hyperparameter Configuration

#### Embedding Model Training
- **Architecture:**  
  - Sentence BERT for multilingual embeddings.
  - Contrastive learning with triplet loss.
- **Training Parameters:**  
  - **Batch Size:** 6 samples.
  - **Learning Rate:** 1e-5 with linear warmup.
  - **Epochs:** 5 epochs with early stopping.

#### BiLSTM Architecture
- **Configuration:**  
  - **Input Dimension:** Matches embedding size.
  - **Hidden Layers:** 2 bidirectional layers.
  - **Hidden Units:** 128–256 per direction.
  - **Dropout Rate:** 0.2–0.3.
  - **Additional:** Batch normalization between layers.

### Hyperparameter Tuning and Model Optimization

- **Optimized Parameters:**  
  - **Learning Rate:** Grid search over [1e-4, 1e-3].
  - **Hidden Layer Size:** Options [128, 256, 512].
  - **Dropout Rates:** Options [0.1, 0.2, 0.3].
  - **Embedding Dimension:** Options [256, 512].
  - **Similarity Threshold:** Options [0.75, 0.80, 0.85].

- **Fine-Tuning Process:**  
  - Utilized the Adam optimizer with weight decay.
  - Implemented learning rate scheduling.
  - Applied gradient clipping at 1.0.
  - Introduced category weighting to address class imbalances.
  - Employed an ensemble of the best-performing models.

To further enhance precision—especially for cases where the cosine similarity score falls below a defined confidence threshold—a Bidirectional LSTM (BiLSTM) rechecker is used. This component reassesses ambiguous cases, leveraging its capacity to capture contextual nuances and ensuring even borderline categorizations are accurately resolved.

### Responsible AI and Ethical Considerations

Our AI development process is firmly grounded in ethical and responsible practices:
- **Ethical Design & Transparency:**  
  - Emphasize fairness, privacy, and transparency.
  - Conduct rigorous evaluations to prevent biases.
- **Robust Evaluation & Continuous Monitoring:**  
  - Implement extensive testing, including manual reviews and advanced verification (e.g., the BiLSTM rechecker).
- **Privacy & Data Security:**  
  - Adhere to strict data governance and industry-leading security standards.
  - Build the system entirely from scratch to prevent data leakage.
- **Commitment to Ongoing Improvement:**  
  - Continuously update models and processes in line with emerging ethical guidelines and technological advancements.

This comprehensive, hybrid approach—integrating LDA, semantic embeddings, cosine similarity, and BiLSTM—delivers superior interpretability, accuracy, and domain-specific adaptability, making it ideally suited for deployment within the National Cybercrime Reporting Portal (NCPR).
