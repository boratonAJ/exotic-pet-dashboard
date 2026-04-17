# Exotic Pet Dashboard

A comprehensive Streamlit-based interactive dashboard for analyzing exotic pet trade data, sentiment analysis, and conservation risk assessment through natural language processing and data visualization.

## Overview

This dashboard provides an integrated platform for researching and analyzing exotic pet-related discussions, comments, and sentiment from various sources. It combines advanced NLP techniques, machine learning models, and interactive visualizations to deliver insights about the exotic pet trade, conservation concerns, and public sentiment.

## Features

### 📊 Data Analysis & Visualization
- **Interactive Plotly Charts**: Dynamic visualizations for trends, distributions, and comparisons
- **Sentiment Analysis**: VADER-based sentiment scoring with positive, negative, and neutral classifications
- **Word Clouds**: Visual representation of frequently mentioned terms and topics
- **Statistical Analysis**: Advanced statistical testing and correlation analysis using Pingouin

### 🔍 Natural Language Processing
- **Wildlife Entity Extraction**: Automatic identification of species names, legal terms, welfare concerns, and conservation keywords
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) for discovering topics in text data
- **Risk Classification**: ML-based classification of conservation and safety risks in comments
- **Text Mining**: CountVectorizer-based analysis of frequently occurring terms and phrases

### 🎯 Conservation Risk Assessment
- **Risk Scoring**: Quantitative scoring system for conservation-related threats
- **Risk Classification**: Categorization into high, medium, and low risk categories
- **Matched Terms Analysis**: Identification of specific terms indicating conservation concerns
- **Trend Tracking**: Historical analysis of risk indicators over time

### 🎨 User Interface
- **Dark Purple Theme**: Custom-styled Streamlit interface with elegant gradient backgrounds
- **Organized Tabs**: Logical grouping of functionality for easy navigation
- **Performance Metrics**: Key statistics and summary metrics displayed prominently
- **Responsive Design**: Optimized for wide displays with max-width layout

## Requirements

### Core Data Science
- numpy 1.24.3
- pandas 2.0.3
- scipy 1.11.2

### Visualization
- matplotlib 3.7.2
- seaborn 0.12.2
- plotly 5.16.1

### Machine Learning & NLP
- scikit-learn 1.3.1
- xgboost 2.0.1
- vaderSentiment 3.3.2
- wordcloud

### Dashboard Framework
- streamlit

### Optional
- pingouin (for advanced statistical analysis)

See [requirements.txt](requirements.txt) for the complete dependency list.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd exotic-pet-dashboard
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   pip list
   ```

## Usage

### Running the Dashboard

```bash
streamlit run combined_streamlit_dashboard_full_themed.py
```

The application will start on `http://localhost:8501` by default.

### Using the Dashboard

1. **Upload or Select Data**: Choose your data source (CSV files from the data folder)
2. **View Sentiment Analysis**: Explore sentiment distributions and trends
3. **Analyze Wildlife Entities**: Identify mentions of species, legal terms, and conservation keywords
4. **Risk Assessment**: Review conservation and safety risk scores
5. **Download Reports**: Export visualizations and data summaries
6. **Export Data**: Download filtered datasets for external analysis

## Project Structure

```
exotic-pet-dashboard/
├── combined_streamlit_dashboard_full_themed.py   # Main dashboard application
├── requirements.txt                               # Python dependencies
├── README.md                                      # This file
└── .git/                                          # Git version control
```

## Key Components

### Data Processing
- **Sentiment Analysis**: Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for robust sentiment scoring
- **Text Vectorization**: CountVectorizer for term frequency analysis
- **Topic Extraction**: LDA for unsupervised topic discovery

### Machine Learning Models
- **Logistic Regression**: For binary/multi-class classification tasks
- **XGBoost**: For advanced predictive modeling
- **Risk Classifier**: Custom conservation risk assessment model

### Visualization Engine
- **Plotly Express**: For interactive, responsive charts
- **Plotly Graph Objects**: For custom chart configurations
- **Matplotlib/WordCloud**: For static visualizations and text analysis

## Data Format

The dashboard expects CSV files with at least one text column containing comments or content to analyze. Optimal structure includes:
- `text` or `comment`: Main content column
- `date`: Timestamp for time-series analysis
- `source`: Origin of the data (optional)
- `category`: Classification or grouping (optional)

## Configuration

### Theme Customization
The dashboard features a dark purple theme with custom CSS styling. To modify the theme:
1. Edit the `apply_global_theme()` function in the main script
2. Update color values in the CSS gradient definitions
3. Restart the Streamlit application

### Model Parameters
- **LDA Topics**: Adjustable number of topics for topic modeling
- **Sentiment Thresholds**: Configurable boundaries for sentiment classifications
- **Risk Scoring**: Customizable weights for risk factor detection

## Troubleshooting

### Issue: Pingouin not available
The dashboard includes fallback functionality if Pingouin is not installed. Advanced statistical analysis will be unavailable, but core functionality remains intact.

### Issue: Slow performance with large datasets
- Reduce dataset size or apply filters before analysis
- Lower the number of LDA topics
- Consider preprocessing text data to remove stop words

### Issue: Memory errors
- Process data in batches
- Reduce text field length
- Close other applications to free system memory

## Performance Considerations

- **Optimal Dataset Size**: < 100,000 rows for smooth performance
- **Text Processing**: Recommended word limit of 500 words per document
- **Visualization**: Plotly charts are interactive but may lag with > 50,000 data points

## Future Enhancements

- [ ] Support for real-time data streaming
- [ ] Advanced NLP models (BERT, transformers)
- [ ] Database integration for persistent storage
- [ ] API endpoint support for external data sources
- [ ] Advanced filtering and query capabilities
- [ ] Custom report generation
- [ ] Multi-language support

## Authors
- Olabode : ola.ajayi@gwu.edu
- Vaishnavi : vaishnavi.pachva@gwmail.gwu.edu
- Henry : Henry.lin@gwmail.gwu.edu
- Connor : c.buss@gwmail.gwu.edu
- Kitty : kittyyangjunbi@gwmail.gwu.edu
- Davide : david.king@gwmail.gwu.edu

## License
See LICENSE file for details.

## Contributing
1. Create a feature branch
2. Make changes and commit
3. Push to remote and create a pull request

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All functions are documented with docstrings
- Existing functionality remains intact
- New features include relevant tests

## License

See the [LICENSE](LICENSE) file for details.

## Contact & Support

For issues, questions, or suggestions regarding this project, please contact the project maintainers or open an issue in the repository.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Sentiment analysis powered by [VADER](https://github.com/cjhutto/vaderSentiment)
- Visualizations using [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- Machine learning with [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/)

---

**Last Updated**: April 2026  
**Version**: 1.0.0
