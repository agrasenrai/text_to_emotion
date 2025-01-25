import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core dependencies
try:
    import pandas as pd
    import numpy as np
    from transformers import pipeline
    import torch
except ImportError as e:
    logger.error(f"Failed to import core dependencies: {str(e)}")
    raise

# Visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    logger.warning(f"Visualization dependencies not available: {str(e)}")
    plt = None
    sns = None

# Social media analysis dependencies
try:
    import tweepy
    from textblob import TextBlob
except ImportError as e:
    logger.warning(f"Social media analysis dependencies not available: {str(e)}")
    tweepy = None
    TextBlob = None

class EmotionClassifier:
    def __init__(self, model_name="bhadresh-savani/distilbert-base-uncased-emotion"):
        """Initialize the emotion classifier with a pre-trained model."""
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
        try:
            self.classifier = pipeline("text-classification", 
                                    model=model_name, 
                                    return_all_scores=True,
                                    device=0 if torch.cuda.is_available() else -1)
            logger.info(f"Model {model_name} loaded successfully")
            logger.info(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for inference")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def classify_text(self, text):
        """Classify the emotion in a given text."""
        try:
            result = self.classifier(text)[0]
            # Get the emotion with highest confidence score
            emotion = max(result, key=lambda x: x['score'])
            return {
                'text': text,
                'emotion': emotion['label'],
                'confidence': emotion['score']
            }
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return None

    def batch_classify(self, texts):
        """Classify emotions for a batch of texts."""
        results = []
        for text in texts:
            result = self.classify_text(text)
            if result:
                results.append(result)
        return results

    def get_emotion_distribution(self, text):
        """Get distribution of all emotions for a given text."""
        try:
            result = self.classifier(text)[0]
            return {score['label']: score['score'] for score in result}
        except Exception as e:
            logger.error(f"Error in emotion distribution analysis: {str(e)}")
            return None

    def analyze_temporal_trends(self, texts, timestamps):
        """Analyze emotion trends over time."""
        try:
            results = []
            for text, timestamp in zip(texts, timestamps):
                emotion_dist = self.get_emotion_distribution(text)
                if emotion_dist:
                    emotion_dist['timestamp'] = timestamp
                    results.append(emotion_dist)
            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return None

    def visualize_emotion_distribution(self, emotions_df, title="Emotion Distribution"):
        """Visualize the distribution of emotions."""
        if plt:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=emotions_df, x='emotion')
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt
        else:
            logger.warning("Matplotlib not available for visualization")
            return None

    def visualize_temporal_trends(self, temporal_df):
        """Visualize emotion trends over time."""
        if plt:
            temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'])
            plt.figure(figsize=(12, 6))
            for emotion in self.emotions:
                plt.plot(temporal_df['timestamp'], temporal_df[emotion], label=emotion)
            plt.title('Emotion Trends Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Emotion Score')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt
        else:
            logger.warning("Matplotlib not available for visualization")
            return None

def analyze_healthcare_sentiments(data_path):
    """Analyze healthcare-related sentiments from a dataset."""
    try:
        # Load and preprocess data
        df = pd.read_csv(data_path)
        classifier = EmotionClassifier()
        
        # Perform classification
        results = classifier.batch_classify(df['text'].tolist())
        
        # Analyze results
        emotions_df = pd.DataFrame(results)
        emotion_stats = emotions_df['emotion'].value_counts()
        
        # Calculate confidence statistics
        confidence_stats = emotions_df.groupby('emotion')['confidence'].agg(['mean', 'std'])
        
        logger.info("Emotion distribution in healthcare texts:")
        logger.info(emotion_stats)
        logger.info("\nConfidence statistics by emotion:")
        logger.info(confidence_stats)
        
        return emotions_df, confidence_stats
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return None, None

class SocialMediaAnalyzer:
    def __init__(self, twitter_api_key=None, twitter_api_secret=None,
                 twitter_access_token=None, twitter_access_token_secret=None):
        """Initialize social media analyzer with API credentials."""
        self.emotion_classifier = EmotionClassifier()
        if all([twitter_api_key, twitter_api_secret, 
                twitter_access_token, twitter_access_token_secret]):
            auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
            auth.set_access_token(twitter_access_token, twitter_access_token_secret)
            self.twitter_api = tweepy.API(auth)
            logger.info("Twitter API initialized successfully")
        else:
            self.twitter_api = None
            logger.warning("Twitter API credentials not provided")

    def analyze_twitter_topic(self, query, count=100):
        """Analyze emotions in tweets about a specific topic."""
        if not self.twitter_api:
            logger.error("Twitter API not initialized")
            return None

        try:
            tweets = self.twitter_api.search_tweets(q=query, lang='en', count=count)
            results = []
            
            for tweet in tweets:
                # Clean tweet text
                clean_text = ' '.join(word for word in tweet.text.split() 
                                    if not word.startswith(('#', '@', 'http')))
                
                # Get emotion analysis
                emotion_result = self.emotion_classifier.classify_text(clean_text)
                if emotion_result:
                    emotion_result['timestamp'] = tweet.created_at
                    emotion_result['likes'] = tweet.favorite_count
                    emotion_result['retweets'] = tweet.retweet_count
                    
                    # Add sentiment polarity using TextBlob
                    sentiment = TextBlob(clean_text).sentiment.polarity
                    emotion_result['sentiment_polarity'] = sentiment
                    
                    results.append(emotion_result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in Twitter analysis: {str(e)}")
            return None

    def visualize_social_insights(self, df):
        """Visualize insights from social media analysis."""
        if df is None or df.empty:
            logger.error("No data available for visualization")
            return

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Emotion distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=df, x='emotion')
        plt.title('Emotion Distribution in Tweets')
        plt.xticks(rotation=45)
        
        # Engagement by emotion
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='emotion', y='likes')
        plt.title('Tweet Engagement by Emotion')
        plt.xticks(rotation=45)
        
        # Sentiment distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=df, x='sentiment_polarity', bins=20)
        plt.title('Sentiment Distribution')
        
        # Temporal trend
        plt.subplot(2, 2, 4)
        df_sorted = df.sort_values('timestamp')
        plt.plot(df_sorted['timestamp'], df_sorted['sentiment_polarity'])
        plt.title('Sentiment Trend Over Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt

if __name__ == "__main__":
    # Example usage with extended analysis
    classifier = EmotionClassifier()
    
    # Sample healthcare-related texts with timestamps
    sample_data = [
        {
            "text": "I'm worried about the increasing COVID-19 cases",
            "timestamp": "2024-01-25 10:00:00"
        },
        {
            "text": "The new treatment has shown promising results",
            "timestamp": "2024-01-25 11:00:00"
        },
        {
            "text": "Healthcare workers are doing an amazing job",
            "timestamp": "2024-01-25 12:00:00"
        },
        {
            "text": "The long waiting times are frustrating",
            "timestamp": "2024-01-25 13:00:00"
        }
    ]
    
    # Basic emotion classification
    print("Basic Emotion Classification:")
    results = classifier.batch_classify([item['text'] for item in sample_data])
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Primary Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Get full emotion distribution
        emotion_dist = classifier.get_emotion_distribution(result['text'])
        print("Emotion Distribution:")
        for emotion, score in emotion_dist.items():
            print(f"  {emotion}: {score:.2f}")
        print()
    
    # Temporal analysis
    print("\nTemporal Emotion Analysis:")
    temporal_df = classifier.analyze_temporal_trends(
        [item['text'] for item in sample_data],
        [item['timestamp'] for item in sample_data]
    )
    if temporal_df is not None:
        print(temporal_df)

    # Add visualization of results
    emotions_df = pd.DataFrame(results)
    
    # Visualize basic emotion distribution
    plt_dist = classifier.visualize_emotion_distribution(emotions_df)
    if plt_dist:
        plt_dist.savefig('emotion_distribution.png')
    
    # Visualize temporal trends
    plt_trends = classifier.visualize_temporal_trends(temporal_df)
    if plt_trends:
        plt_trends.savefig('temporal_trends.png')
    
    # Example of social media analysis (requires Twitter API credentials)
    twitter_credentials = {
        'twitter_api_key': 'YOUR_API_KEY',
        'twitter_api_secret': 'YOUR_API_SECRET',
        'twitter_access_token': 'YOUR_ACCESS_TOKEN',
        'twitter_access_token_secret': 'YOUR_ACCESS_TOKEN_SECRET'
    }
    
    social_analyzer = SocialMediaAnalyzer(**twitter_credentials)
    
    # Analyze tweets about healthcare
    tweet_analysis = social_analyzer.analyze_twitter_topic(
        query="healthcare OR medical OR hospital",
        count=100
    )
    
    if tweet_analysis is not None:
        # Visualize social media insights
        plt_social = social_analyzer.visualize_social_insights(tweet_analysis)
        if plt_social:
            plt_social.savefig('social_media_insights.png')
        
        print("\nSocial Media Analysis Summary:")
        print(f"Total tweets analyzed: {len(tweet_analysis)}")
        print("\nEmotion distribution in tweets:")
        print(tweet_analysis['emotion'].value_counts())
        print("\nAverage engagement by emotion:")
        print(tweet_analysis.groupby('emotion')['likes'].mean())
