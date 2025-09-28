import random

class HashtagGenerator:
    def __init__(self):
        """
        Initializes the Hashtag Generator.
        In a real-world application, this could connect to a service to get trending tags.
        For this example, we'll use predefined lists.
        """
        self.trending_tags = [
            '#explorepage', '#instagood', '#photography', '#picoftheday',
            '#photooftheday', '#instadaily', '#love', '#beautiful', '#art'
        ]

    def _clean_tag(self, text: str) -> str:
        """Removes spaces and special characters to create a valid hashtag."""
        return ''.join(char for char in text if char.isalnum())

    def generate_hashtags(self, description: str, objects: list, themes: list) -> dict:
        """
        Generates a comprehensive set of hashtags based on image content.

        Args:
            description (str): A detailed description of the image from the LLM.
            objects (list): A list of detected objects.
            themes (list): A list of identified themes.

        Returns:
            dict: Categorized hashtags.
        """
        # 1. High-Reach Tags (Trending + Broad Themes)
        high_reach = [f"#{self._clean_tag(theme)}" for theme in themes]
        high_reach.extend(random.sample(self.trending_tags, min(len(self.trending_tags), 5)))
        
        # 2. Niche Tags (Specific Objects)
        niche = [f"#{self._clean_tag(obj)}" for obj in objects]
        
        # 3. Descriptive Long-Tail Tags (from the detailed description)
        descriptive = set()
        # A simple keyword extraction from the description
        keywords = [word for word in description.lower().split() if len(word) > 4 and word.isalpha()]
        
        # Combine objects and themes with context
        if 'person' in objects or 'people' in objects:
            descriptive.add("#PortraitPhotography")
        if 'nature' in themes or 'mountain' in description:
            descriptive.add("#NatureLovers")
            descriptive.add("#LandscapePhotography")
        if 'city' in themes or 'urban' in themes:
            descriptive.add("#Cityscape")
            descriptive.add("#UrbanPhotography")
        
        # Add a few keywords as tags
        for kw in random.sample(keywords, min(len(keywords), 3)):
             descriptive.add(f"#{self._clean_tag(kw)}")
        
        return {
            "high_reach": list(set(high_reach)),
            "niche": list(set(niche)),
            "descriptive": list(descriptive)
        }

# import random

# class HashtagGenerator:
#     def __init__(self):
#         """
#         Initializes the Hashtag Generator with predefined popular tags.
#         """
#         self.popular_tags = [
#             '#explorepage', '#instagood', '#photography', '#picoftheday',
#             '#photooftheday', '#instadaily', '#love', '#beautiful', '#art',
#             '#style', '#happy', '#follow', '#cute', '#travel'
#         ]

#     def _clean_tag(self, text: str) -> str:
#         """Removes spaces and special characters to create a valid hashtag."""
#         return ''.join(char for char in text if char.isalnum())

#     def generate_hashtags(self, description: str, objects: list, themes: list, mood: str) -> dict:
#         """
#         Generates a comprehensive set of hashtags based on image content and mood.

#         Args:
#             description (str): A detailed description of the image.
#             objects (list): A list of detected objects.
#             themes (list): A list of identified themes.
#             mood (str): The dominant mood detected in the image (e.g., 'Happy', 'Sad').

#         Returns:
#             dict: Categorized hashtags: 'popular', 'specific', and 'expressive'.
#         """
#         # 1. Popular Tags (Trending + Broad Themes)
#         popular = [f"#{self._clean_tag(theme)}" for theme in themes]
#         popular.extend(random.sample(self.popular_tags, min(len(self.popular_tags), 5)))
        
#         # 2. Specific Tags (Detected Objects)
#         specific = [f"#{self._clean_tag(obj)}" for obj in objects]
        
#         # 3. Expressive Tags (Mood-based and Creative)
#         expressive = set()
#         mood_lower = mood.lower()

#         # Add tags based on the detected mood
#         if 'happy' in mood_lower or 'joy' in mood_lower:
#             expressive.update(['#GoodVibes', '#HappyMoments', '#MakingMemories', '#SmileMore'])
#         elif 'sad' in mood_lower:
#             expressive.update(['#MoodyGrams', '#DeepThoughts', '#Reflection', '#ItsOkayNotToBeOkay'])
#         elif 'surprise' in mood_lower:
#             expressive.update(['#Unexpected', '#WowMoment', '#Surprise'])
#         elif 'neutral' in mood_lower:
#              expressive.update(['#Chillin', '#RelaxedVibes', '#SimpleThings'])
        
#         # Add context-based creative tags
#         if 'person' in objects or 'people' in objects:
#             expressive.add("#FriendshipGoals")
#         if 'nature' in themes or 'mountain' in description or 'beach' in description:
#             expressive.add("#NatureLovers")
#             expressive.add("#GetOutside")
#         if 'city' in themes or 'urban' in themes:
#             expressive.add("#CityLife")
#             expressive.add("#StreetPhotography")

#         return {
#             "Popular": list(set(popular)),
#             "Specific": list(set(specific)),
#             "Expressive": list(expressive)
#         }

