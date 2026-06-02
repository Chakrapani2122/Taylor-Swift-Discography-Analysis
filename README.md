# Taylor Swift Discography Analysis

**Comprehensive Lyrical Analysis of Taylor Swift's Discography: Trends, Themes, and Sentiment**

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Data Description](#data-description)
- [Methodologies](#methodologies)
- [Dependencies & Technologies](#dependencies--technologies)
- [How to Contribute](#how-to-contribute)
- [License](#license)

---

## Overview

This project conducts a **comprehensive data-driven analysis** of Taylor Swift's complete discography by integrating web scraping, text analytics, and sentiment analysis. By systematically collecting metadata and lyrical content for every track—including song titles, album names, release dates, durations, streaming play counts, and full lyrics—the project builds a unified dataset that supports both quantitative trend analysis and deeper linguistic evaluation.

The analysis leverages machine learning and NLP techniques to uncover:
- **Artistic evolution** across different eras and albums
- **Thematic patterns** and recurring lyrical motifs
- **Emotional tone** progression through sentiment analysis
- **Audience engagement** metrics through streaming statistics
- **Distinctive vocabulary** and linguistic characteristics

### Value Proposition

This analysis produces actionable insights for:
- **Record labels and marketing teams** seeking data-driven promotion strategies
- **Producers and songwriters** analyzing thematic and stylistic patterns
- **Streaming platforms and media analysts** evaluating text-driven recommendation potential
- **Artist management teams** exploring long-term creative evolution
- **Fans and music critics** interested in deeper lyrical interpretation and artist analysis

---

## Project Objectives

### 1. Dataset Creation
- Develop a complete, structured dataset covering track-level and album-level attributes
- Scrape full lyrics for every song in the complete catalog
- Capture key numerical features such as release year, runtime, and streaming play counts
- Build a persistent database enabling incremental scraping across multiple sessions

### 2. Quantitative Trend Analysis
- Analyze track release frequency by year and across albums
- Examine song duration patterns and evolution
- Evaluate play counts to infer listener engagement and career trajectory
- Track statistical trends across different eras

### 3. Text Analytics
- Apply word frequency analysis to identify dominant vocabulary
- Compute TF (Term Frequency) and TF-IDF metrics to isolate distinctive words
- Extract bigrams to uncover repeated phrases and thematic expressions
- Conduct part-of-speech tagging to focus on nouns, verbs, adjectives, and narrative elements
- Perform detailed linguistic analysis of song structures

### 4. Sentiment Analysis
- Assess emotional tone across songs and different albums
- Compare lexicon-based (Bing-Liu dictionary) and ML-based (TextBlob) sentiment scoring approaches
- Track sentiment trends by album and over time
- Identify shifts in emotional expression across Taylor's artistic eras

### 5. Topic Modeling
- Apply multiple topic modeling frameworks: **LDA**, **NMF**, and **BERTopic**
- Determine the optimal number of topics using coherence and perplexity analysis
- Visualize topic distributions, cluster relationships, and representative words
- Extract interpretable themes characterizing different eras of Taylor Swift's songwriting

---

## Key Features

✨ **Comprehensive Data Collection**
- Web scraping from Spotify using Selenium WebDriver
- Complete lyrics extraction and storage
- Metadata collection including play counts and release information
- Duplicate detection and incremental database updates

📊 **Advanced Text Analytics**
- Word frequency analysis (TF, TF-IDF, bigrams)
- Part-of-speech tagging for thematic analysis
- Token normalization and text preprocessing
- Vocabulary profiling across albums and eras

💭 **Multi-Method Sentiment Analysis**
- Bing-Liu lexicon-based sentiment classification
- TextBlob continuous sentiment polarity scoring
- Comparative analysis of sentiment methodologies
- Era-based sentiment trend tracking

🏷️ **Topic Discovery & Modeling**
- LDA (Latent Dirichlet Allocation) topic modeling
- NMF (Non-negative Matrix Factorization)
- BERTopic transformer-based topic extraction
- Interactive visualizations and cluster analysis

📈 **Rich Visualization Capabilities**
- Word cloud generation for frequency visualization
- Topic distribution charts and timeline analysis
- Sentiment progression graphs
- Album-level comparative analytics

---

## Installation & Setup

### Prerequisites

Before starting, ensure you have:
- **Python 3.8 or higher** installed on your system
- **pip** (Python package manager)
- **Jupyter Notebook** for running the analysis
- **Google Chrome** browser (for web scraping with Selenium)
- **ChromeDriver** (automatically managed by webdriver_manager)
- Minimum **2GB RAM** for data processing and modeling
- **Internet connection** for web scraping and downloading NLP resources

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chakrapani2122/Taylor-Swift-Discography-Analysis.git
cd Taylor-Swift-Discography-Analysis
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not available, install packages manually:

```bash
# Core data processing
pip install pandas numpy

# Web scraping
pip install selenium webdriver-manager

# Natural Language Processing
pip install nltk textblob

# Machine learning and NLP
pip install scikit-learn gensim bertopic

# Visualization
pip install matplotlib seaborn wordcloud pyLDAvis

# Optional: for enhanced notebook experience
pip install jupyter jupyterlab
```

### Step 4: Download Required NLP Resources

```bash
python3 << 'EOF'
import nltk

# Download required NLTK resources
resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']
for resource in resources:
    nltk.download(resource)
    print(f"Downloaded: {resource}")
EOF
```

### Step 5: Verify Installation

```bash
python3 << 'EOF'
import pandas as pd
import nltk
from selenium import webdriver
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation

print("✓ All core libraries installed successfully!")
EOF
```

### Step 6: Launch Jupyter Notebook

```bash
jupyter notebook Project.ipynb
```

This opens the analysis notebook in your default web browser. You can now run the project step-by-step.

---

## Project Structure

```
Taylor-Swift-Discography-Analysis/
│
├── README.md                          # This file - complete project documentation
├── LICENSE                            # Project license file
├── Project.ipynb                      # Main Jupyter notebook with full analysis pipeline
│
├── Data Files (Input & Intermediate):
│   ├── songs.csv                      # Spotify song URLs (for web scraping reference)
│   ├── songsdetails.csv               # Raw scraped song metadata and lyrics
│   └── cleanedsongs.csv               # Cleaned and processed song dataset
│
├── Sentiment Lexicons (for analysis):
│   ├── Bing_liu_positive.txt          # Positive sentiment words dictionary (2,005 words)
│   └── Bing_liu_negative.txt          # Negative sentiment words dictionary (4,782 words)
│
└── .gitattributes                     # Git configuration for file handling
```

### File Descriptions

#### **Project.ipynb** - Main Analysis Notebook
The complete Jupyter notebook containing the entire analysis pipeline:
- **Section 1-2**: Environment setup and library imports
- **Section 3**: Web scraping from Spotify
- **Section 4**: Data cleaning and feature engineering
- **Section 5**: Exploratory data analysis (EDA)
- **Section 6**: Text analytics (word frequency, TF-IDF, bigrams, POS tagging)
- **Section 7**: Sentiment analysis (Bing-Liu and TextBlob methods)
- **Section 8**: Topic modeling (LDA, NMF, BERTopic)
- **Section 9**: Visualizations and insights

#### **CSV Data Files**

| File | Purpose | Size | Records |
|------|---------|------|---------|
| `songs.csv` | Song URLs from Spotify (for scraping reference) | 11 KB | ~250 songs |
| `songsdetails.csv` | Raw scraped data with full lyrics | 721 KB | ~250 songs with metadata |
| `cleanedsongs.csv` | Cleaned/processed data ready for analysis | 384 KB | ~250 songs (deduplicated) |

#### **Sentiment Lexicon Files**

- `Bing_liu_positive.txt`: 2,005 positive sentiment words
- `Bing_liu_negative.txt`: 4,782 negative sentiment words
- **Source**: Bing Liu's sentiment lexicon, widely used in text analysis

---

## Usage Guide

### Running the Complete Analysis

1. **Open the notebook** in Jupyter:
   ```bash
   jupyter notebook Project.ipynb
   ```

2. **Execute cells sequentially**:
   - Use `Shift+Enter` to run each cell
   - Follow the markdown instructions within the notebook
   - Monitor output messages for any warnings or errors

3. **Review intermediate outputs**:
   - The notebook generates CSV files at key stages
   - Check the `./` directory to see newly created/updated files

### Key Analysis Sections

#### A. Web Scraping (Section 3)
Extract song metadata from Spotify:
```
Input: Spotify Taylor Swift artist page
Process: Selenium automation to scrape URLs and song details
Output: songs.csv, songsdetails.csv
```

**Note**: First-time scraping takes 30-60 minutes due to API rate limits and dynamic content loading.

#### B. Data Cleaning (Section 4)
Prepare raw data for analysis:
```
Input: songsdetails.csv (raw scraped data)
Process: Remove duplicates, clean text, parse duration, normalize play counts
Output: cleanedsongs.csv (analysis-ready dataset)
```

#### C. Text Analytics (Section 6)
Analyze vocabulary and linguistic patterns:
```python
# Example: Analyze word frequency in a specific album
album_lyrics = cleaned_data[cleaned_data['AlbumName'] == 'folklore']['Lyrics'].str.cat()
word_freq = analyze_word_frequency(album_lyrics)
visualize_word_cloud(word_freq)
```

#### D. Sentiment Analysis (Section 7)
Examine emotional tone:
```python
# Example: Calculate sentiment for each album
sentiment_by_album = data.groupby('AlbumName').apply(
    lambda x: calculate_sentiment(x['Lyrics'].str.cat())
)
plot_sentiment_trends(sentiment_by_album)
```

#### E. Topic Modeling (Section 8)
Discover thematic patterns:
```python
# Example: Extract topics using BERTopic
from bertopic import BERTopic
topic_model = BERTopic(language="english", min_topic_size=5)
topics, probabilities = topic_model.fit_transform(documents)
topic_model.visualize_topics()
```

### Command-Line Examples

**Extract and update song database**:
```bash
python3 << 'EOF'
import pandas as pd
from Project import scrape_spotify_song_urls

# Load existing data
existing_urls = set(pd.read_csv('songs.csv')['link'])

# Scrape new songs
scrape_spotify_song_urls(browser, existing_urls)
EOF
```

**Perform sentiment analysis on cleaned data**:
```bash
python3 << 'EOF'
import pandas as pd
from textblob import TextBlob

data = pd.read_csv('cleanedsongs.csv')
data['Sentiment'] = data['Lyrics'].apply(
    lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0
)
print(data.groupby('AlbumName')['Sentiment'].mean().sort_values(ascending=False))
EOF
```

---

## Data Description

### Dataset Overview
- **Total Songs**: ~250 (complete Taylor Swift discography)
- **Data Points**: Song name, album, release year, duration, play count, complete lyrics
- **Time Period**: ~2008-2025 (all released eras)
- **Data Source**: Spotify (via web scraping)

### Column Definitions (cleanedsongs.csv)

| Column | Type | Description |
|--------|------|-------------|
| `SongName` | String | Official song title |
| `AlbumName` | String | Album the song belongs to |
| `Year` | Integer | Release year |
| `Duration` | Integer | Song length in seconds |
| `PlayCount` | Integer | Total Spotify play count |
| `Lyrics` | String | Complete song lyrics |
| `SongLink` | String | Spotify song URL |
| `AlbumLink` | String | Spotify album URL |

### Data Quality Notes
- **Lyrics**: Full lyrics extracted from Spotify; some songs may have incomplete data if unavailable
- **Play Count**: Represents snapshot at scraping time; changes continuously
- **Duration**: Converted to seconds for numerical analysis
- **Duplicates**: Removed (remixes, live versions, deluxe editions treated separately)

---

## Methodologies

### 1. Web Scraping with Selenium
**Purpose**: Collect comprehensive song metadata and lyrics from Spotify

**Technical Approach**:
- Use Selenium WebDriver with Chrome automation
- Bypass automation detection using specialized Chrome options
- Handle dynamic content loading with implicit waits
- Implement duplicate detection to avoid redundant requests
- Save progress to CSV for fault tolerance

**Advantages**:
- Captures real-time play count data
- Obtains complete lyrics (when available)
- Handles JavaScript-heavy dynamic content
- Enables incremental updates

### 2. Text Analytics
**Purpose**: Extract meaningful linguistic patterns and vocabulary insights

**Techniques Applied**:

**a) Word Frequency Analysis**
```
Process: Count occurrences of each word in corpus
Metric: TF (Term Frequency) = word_count / total_words
Use Case: Identify dominant themes and vocabulary
```

**b) TF-IDF (Term Frequency-Inverse Document Frequency)**
```
Formula: TF-IDF = TF × log(total_docs / docs_containing_term)
Use Case: Find distinctive words specific to each song/album
Insight: High TF-IDF words are topic-specific and discriminative
```

**c) Bigram Analysis**
```
Process: Extract pairs of consecutive words
Use Case: Uncover recurring phrases and lyrical patterns
Example: "Love me", "Never letting go", "Shake it off"
```

**d) Part-of-Speech (POS) Tagging**
```
Tools: NLTK averaged_perceptron_tagger
Process: Classify each word as noun, verb, adjective, etc.
Use Case: Focus on nouns/adjectives for thematic analysis
Example: Extract all adjectives to understand emotional vocabulary
```

### 3. Sentiment Analysis
**Purpose**: Quantify emotional tone and track sentiment progression

**Method A: Lexicon-Based (Bing-Liu)**
```
Process: Compare words against positive/negative dictionaries
Scoring: (Positive_words - Negative_words) / Total_words
Advantages: Interpretable, no training needed
Limitations: Domain-specific words may be misclassified
```

**Method B: Machine Learning-Based (TextBlob)**
```
Algorithm: Pre-trained polarity classifier
Range: -1.0 (negative) to +1.0 (positive)
Advantages: Captures context and negations
Limitations: Black-box approach, less interpretable
```

**Comparison**:
- Run both methods to cross-validate results
- Aggregate scores at song and album levels
- Analyze trends over time

### 4. Topic Modeling
**Purpose**: Discover latent themes and topics in lyrical content

**Method A: Latent Dirichlet Allocation (LDA)**
```
Algorithm: Probabilistic topic model
Output: Topics (word distributions) and document-topic assignments
Configuration: num_topics (typically 5-15), passes=10, workers=4
Evaluation: Coherence score and perplexity
Advantages: Widely used, interpretable
```

**Method B: Non-negative Matrix Factorization (NMF)**
```
Algorithm: Matrix decomposition with non-negativity constraints
Output: Topic matrix (words) and document matrix (topic weights)
Configuration: n_components (number of topics), init='random'
Advantages: Faster than LDA, suitable for sparse data
```

**Method C: BERTopic**
```
Algorithm: Transformer-based semantic topic modeling
Process: BERT embeddings → UMAP dimensionality reduction → HDBSCAN clustering
Output: Semantically meaningful topics with representative documents
Advantages: State-of-the-art, captures semantic meaning
```

---

## Dependencies & Technologies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥1.0 | Data manipulation and CSV handling |
| **numpy** | ≥1.18 | Numerical computations |
| **nltk** | ≥3.5 | Tokenization, POS tagging, stopwords |
| **textblob** | ≥0.15 | Sentiment analysis |
| **selenium** | ≥3.0 | Web scraping browser automation |
| **webdriver-manager** | ≥3.0 | Automatic ChromeDriver management |
| **scikit-learn** | ≥0.24 | TF-IDF, LDA, NMF algorithms |
| **gensim** | ≥4.0 | Advanced topic modeling |
| **bertopic** | ≥0.10 | Transformer-based topic modeling |
| **matplotlib** | ≥3.0 | Static visualizations |
| **seaborn** | ≥0.11 | Statistical data visualization |
| **wordcloud** | ≥1.8 | Word frequency visualizations |
| **pyLDAvis** | ≥3.2 | Interactive topic model visualization |

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.8+
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Disk Space**: ~1GB for dependencies and data
- **Browser**: Google Chrome (for web scraping)

### Optional Tools
- **Jupyter Notebook/Lab**: For interactive analysis
- **Git**: For version control
- **VS Code**: Recommended code editor

---

## How to Contribute

We welcome contributions to improve this analysis! Here's how you can help:

### Types of Contributions

1. **Bug Reports**
   - Open an issue describing the problem
   - Include error messages and steps to reproduce
   - Specify your Python version and OS

2. **Feature Enhancements**
   - Suggest new analysis techniques or visualizations
   - Propose data collection improvements
   - Recommend additional NLP methods

3. **Code Improvements**
   - Optimize existing algorithms
   - Improve code documentation and clarity
   - Add error handling and validation

4. **Documentation**
   - Fix typos or clarify explanations
   - Add more usage examples
   - Improve readability

### Contribution Workflow

1. **Fork** the repository on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** thoroughly
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Submit a Pull Request** with a detailed description

### Coding Standards
- Follow PEP 8 Python style guidelines
- Add docstrings to new functions
- Include comments for complex logic
- Test code before submitting

### Questions or Discussions?
Feel free to open an issue for:
- Questions about the project
- Ideas for improvements
- Discussion of results and insights

---

## License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

### MIT License Summary
- ✅ **Permitted**: Commercial use, modification, distribution, private use
- ❌ **Forbidden**: Liability, warranty
- ℹ️ **Required**: License and copyright notice

For the full license text, see: [LICENSE](LICENSE)

### Attribution
When using this project, please provide attribution to:
- **Author**: Chakrapani Gajji
- **Course**: MIS 670 - Social Media Analytics & Web Mining
- **Original Project**: Taylor Swift Discography Analysis

---

## Project Information

| Item | Details |
|------|---------|
| **Project Type** | Data Analysis & NLP |
| **Primary Language** | Python 3.8+ |
| **Notebook Type** | Jupyter Notebook |
| **Data Source** | Spotify Web Scraping |
| **Analysis Focus** | Lyrical patterns, sentiment, topics |
| **Course** | MIS 670 - Social Media Analytics & Web Mining |
| **Author** | Chakrapani Gajji |
| **Last Updated** | December 2025 |

---

## Troubleshooting

### Common Issues

**Issue**: "ChromeDriver not found"
```
Solution: Run `pip install --upgrade webdriver-manager`
The library automatically downloads the correct ChromeDriver
```

**Issue**: "NLTK resource not found"
```
Solution: Run the NLTK download script in Setup Step 4
Ensure all resources are downloaded: punkt, stopwords, wordnet, averaged_perceptron_tagger
```

**Issue**: "Selenium TimeoutException during scraping"
```
Solution: Increase the implicit wait timeout in Project.ipynb Section 3
Modify: browser.implicitly_wait(30) → browser.implicitly_wait(60)
```

**Issue**: "Memory error during topic modeling"
```
Solution: Reduce the number of documents or use BERTopic instead of LDA
For large datasets, process documents in batches
```

**Issue**: "Spotify access blocked or rate limited"
```
Solution: Add delays between requests
Modify scraping batch size or wait times
Consider using a VPN if access is restricted
```

---

## Resources & References

### Python NLP Libraries
- [NLTK Documentation](https://www.nltk.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)

### Web Scraping
- [Selenium Documentation](https://www.selenium.dev/)
- [webdriver-manager](https://github.com/SergeyPirogov/webdriver_manager)

### Data Analysis
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)

### Visualization
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [pyLDAvis Documentation](https://pyldavis.readthedocs.io/)

---

## Acknowledgments

- **Spotify** for providing comprehensive music data
- **NLTK Project** for NLP resources
- **Open-source community** for excellent Python libraries
- **Bing Liu** for the sentiment lexicon

---

## Contact & Support

For questions, issues, or suggestions:
1. Open an issue on GitHub
2. Check existing issues for solutions
3. Review the troubleshooting section above
4. Consult the original notebook for detailed explanations

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Status**: Complete and Functional
