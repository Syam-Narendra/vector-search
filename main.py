from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy
import faiss

loader = PlaywrightURLLoader(urls=["https://en.wikipedia.org/wiki/Realme"])

data = loader.load()
data = data[0].page_content
print(type(data))

splititer = RecursiveCharacterTextSplitter(
    separators=["\n", "\n\n", ". "], chunk_size=200, chunk_overlap=0
)
# chuncks = numpy.array(splititer.split_text(data))
chuncks = [
    "Meditation and yoga can improve mental health	Health",
    "1	Fruits whole grains and vegetables helps control blood pressure	Health",
    "2	These are the latest fashion trends for this week	Fashion",
    "3	Vibrant color jeans for male are becoming a trend	Fashion",
    "4	The concert starts at 7 PM tonight	Event",
    "5	Navaratri dandiya program at Expo center in Mumbai this october	Event",
    "6	Exciting vacation destinations for your next trip	Travel",
    "7	Maldives and Srilanka are gaining popularity in terms of low budget vacation places	Travel",
]
print(len(chuncks))
encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(chuncks)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
sQuery = "dusshera is a 9 day festivals"
sQVector = numpy.array(encoder.encode(sQuery)).reshape(1, -1)
resultVector, I = index.search(sQVector, k=3)
res = I.tolist()
# fileterSearch = chuncks.loc
fir, sec, third = res[0][0], res[0][1], res[0][2]
print(chuncks[fir])
print(chuncks[sec])
print(chuncks[third])
