from nltk.corpus import twitter_samples
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy
import random

def getFeatureList(tweets, tone):
	arr = []
	for tweet in tweets:
		arr.append((extractFeatures(tweet), tone))
	return arr

def extractFeatures(sentence):
	obj = {}
	for word in sentence.split():
		obj[word] = True
	return obj


positive_tweets = getFeatureList(twitter_samples.strings("positive_tweets.json"), "pos")
negative_tweets = getFeatureList(twitter_samples.strings("negative_tweets.json"), "neg")

data = positive_tweets + negative_tweets
random.shuffle(data)

test_data = data[:8000]
train_data = data[8000:]

classifier = NaiveBayesClassifier.train(test_data)
accuracy_rate = accuracy(classifier, test_data)
print("Accuracy: ", accuracy_rate)

test_tweets = [
	"Just got my first promotion at work! So happy! 🎉 #Blessed",
	"Stuck in traffic for hours. Worst day ever. 😡 #Frustrated",
	"What a beautiful sunset! Feeling grateful. 🌅 #NatureLove",
	"Lost my wallet today. Could this day get any worse? 😞 #Annoyed",
	"Met the kindest person today. Restores my faith in humanity. ❤️ #GoodVibes",
	"Missed my flight because of airport delays. Ugh! #TravelFail",
	"The concert last night was AMAZING! Best night ever! 🎶 #MusicLover",
	"Burnt my dinner... again. I'm hopeless. 😔 #KitchenDisaster",
	"Finally finished my marathon training! Feeling unstoppable! 🏃‍♂️ #Goals",
	"Got scammed online. People can be so cruel. #NeverAgain",
	"Weekend getaway was everything I needed. Refreshing! 🏖️ #Relaxation",
	"Spilled coffee all over my laptop. 😭 #MondayBlues",
	"Adopted a puppy today! Can't wait for new adventures together. 🐶 #DogLover",
	"Another sleepless night. Insomnia is the worst. 😩 #TiredAF",
	"Got accepted into my dream college! So excited for this journey! 🎓 #Achievement",
	"Forgot my umbrella and got drenched. Typical me. ☔ #Unlucky",
	"Had the best time catching up with old friends. 💕 #Grateful",
	"Dropped my phone in the toilet. Just my luck. 😒 #Clumsy",
	"My favorite team won the championship! What a game! 🏆 #Victory",
	"Overcharged at the store. Why does this keep happening to me? 😡 #Annoyed",
	"Discovered a new cafe today. Their coffee is divine! ☕ #Foodie",
	"Customer service these days is just terrible. So disappointed. #NeverAgain",
	"Got a surprise package from an old friend. Made my day! 🎁 #Unexpected",
	"Car broke down in the middle of nowhere. I'm so done. 🚗💔 #Stranded",
	"Scored the winning goal today! Still can't believe it! ⚽ #Champion",
	"Neighbors threw a loud party all night. Zero sleep. 😠 #Rude",
	"Baked my first cake today, and it turned out great! 🎂 #Success",
	"Lost another game in a row. Feeling defeated. 🎮 #GamingFail",
	"Went hiking today. The views were breathtaking! 🏞️ #Adventure",
	"Accidentally sent an email to the wrong person. Embarrassed! #WorkFail",
	"Saw a double rainbow today! Nature is so magical! 🌈 #Lucky",
	"Overpaid for parking again. City life sucks sometimes. #RipOff",
	"Won a gift card in a random contest! Free coffee for a week! ☕ #Winner",
	"Delivery was late, and the food was cold. Never ordering here again. 🍕 #Disappointed",
	"Found $20 in an old jacket. What a great start to the day! 💵 #Fortune",
	"Wi-Fi down for hours. Can't get any work done. 😤 #FirstWorldProblems",
	"Got a compliment from a stranger today. Small things matter. 💫 #FeelGood",
	"Printer jammed during an important deadline. Just why? 😡 #OfficeIssues",
	"Tried a new workout and loved it! Feeling so energized. 💪 #Fitness",
	"Burnt my toast again. How hard is breakfast? 😒 #MorningFails",
	"Sunshine and clear skies today! Perfect weather for a picnic. 🌞 #HappyDay",
	"Flat tire on the way to work. Late again. 🚗😤 #BadLuck",
	"Learned something new today. Always a good feeling! 📚 #Growth",
	"Cancelled plans again. Why do people flake so much? 😠 #Frustrated",
	"Donated to a charity today. Feels great to give back! 💖 #GoodKarma",
	"Rain ruined my outdoor plans. So disappointed. 🌧️ #Unlucky",
	"Discovered a new book series I can't put down! 📖 #Bookworm",
	"Forgot my password. Locked out of my account. Ugh. 😒 #TechIssues",
	"Planted my first garden today! Excited to see it grow. 🌱 #Hobby",
	"Overslept and missed an important meeting. What a disaster. ⏰ #Regret",
]

for tweet in test_tweets:
	tweet_words = extractFeatures(tweet)
	prediction = classifier.classify(tweet_words)
	print(f"Prediction for {tweet}:", prediction)