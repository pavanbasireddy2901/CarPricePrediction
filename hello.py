# Install necessary packages
# Run this command in your terminal or command prompt:
# pip install openai pinecone-client

##import pinecone

# Variable definitions
OPENAI_API_KEY = 'your-openai-api-key'
PINECONE_API_KEY = 'your-pinecone-api-key'
PINECONE_CLOUD = 'your-pinecone-cloud'
PINECONE_INDEX_NAME = 'chat-history'
MAX_TOKENS = 225

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_CLOUD)

# Define the conversation history
history = [
    "1: User: Hi there! How are you doing today? | Bot: Hello! I'm doing great, thank you! How can I assist you today?",
    "2: User: What's the weather like today in New York? | Bot: Today in New York, it's sunny with a slight chance of rain.",
    "3: User: Great! Do you have any good lunch suggestions? | Bot: Sure! How about trying a new salad recipe?",
    "4: User: That sounds healthy. Any specific recipes? | Bot: You could try a quinoa salad with avocado and chicken.",
    "5: User: Sounds delicious! I'll try it. What about dinner? | Bot: For dinner, you could make grilled salmon with vegetables.",
    "6: User: Thanks for the suggestions! Any dessert ideas? | Bot: How about a simple fruit salad or yogurt with honey?",
    "7: User: Perfect! Now, what are some good exercises? | Bot: You can try a mix of cardio and strength training exercises.",
    "8: User: Any specific recommendations for cardio? | Bot: Running, cycling, and swimming are all excellent cardio exercises.",
    "9: User: I'll start with running. Can you recommend any books? | Bot: 'Atomic Habits' by James Clear is a highly recommended book.",
    "10: User: I'll check it out. What hobbies can I take up? | Bot: You could explore painting, hiking, or learning a new instrument.",
    "11: User: Hiking sounds fun! Any specific trails? | Bot: There are great trails in the Rockies and the Appalachian Mountains.",
    "12: User: I'll plan a trip. What about indoor activities? | Bot: Indoor activities like reading, cooking, or playing board games.",
    "13: User: Nice! Any good board games? | Bot: Settlers of Catan and Ticket to Ride are both excellent choices.",
    "14: User: I'll try them out. Any movie recommendations? | Bot: 'Inception' and 'The Matrix' are must-watch movies.",
    "15: User: I love those movies! Any TV shows? | Bot: 'Breaking Bad' and 'Stranger Things' are very popular.",
    "16: User: Great choices! What about podcasts? | Bot: 'How I Built This' and 'The Daily' are very informative.",
    "17: User: Thanks! What are some good travel destinations? | Bot: Paris, Tokyo, and Bali are amazing travel spots.",
    "18: User: I'll add them to my list. Any packing tips? | Bot: Roll your clothes to save space and use packing cubes.",
    "19: User: That's helpful! What about travel insurance? | Bot: Always get travel insurance for safety and peace of mind.",
    "20: User: Thanks for the tips! Any last advice? | Bot: Just enjoy your journey and make the most out of your experiences."
]

# Define the function to add embeddings to Pinecone
def add_embeddings_to_pinecone(history, index_name=PINECONE_INDEX_NAME):
    # Create Pinecone index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536, metric='cosine')

    # Encode each message in the history using OpenAI's model
    embeddings = []
    for i, message in enumerate(history):
        # Use OpenAI's model to encode the message
        response = openai.Embedding.create(input=message, model='text-embedding-ada-002')
        embedding = response['data'][0]['embedding']
        embeddings.append((str(i), embedding))

    # Upsert the encoded messages into the Pinecone index
    index = pinecone.Index(index_name)
    index.upsert(vectors=embeddings)

# Define the RAG mechanism
def retrieve_relevant_history(query, index_name=PINECONE_INDEX_NAME):
    # Encode the query using OpenAI's model
    response = openai.Embedding.create(input=query, model='text-embedding-ada-002')
    query_embedding = response['data'][0]['embedding']
    
    # Query the Pinecone index with the encoded query to retrieve the most relevant messages
    index = pinecone.Index(index_name)
    results = index.query(queries=[query_embedding], top_k=3)
    
    return results

# Function to prepare the prompt
def prepare_prompt(test_prompt, index_name=PINECONE_INDEX_NAME):
    # Retrieve relevant history messages using the RAG mechanism
    relevant_results = retrieve_relevant_history(test_prompt, index_name)
    
    # Combine the retrieved messages with the test prompt
    combined_prompt = ""
    context_referred = []
    for match in relevant_results['matches']:
        # Extract the text of the matched message
        match_id = match['id']
        if match_id.isdigit():
            context_referred.append(history[int(match_id)])
        combined_prompt += history[int(match_id)] + " "
    
    combined_prompt += test_prompt
    
    # Ensure the combined prompt does not exceed the maximum token limit
    if len(combined_prompt) > MAX_TOKENS:
        combined_prompt = combined_prompt[:MAX_TOKENS]
    
    return combined_prompt, context_referred

# Function to test the prompt
def test_final_prompt():
    final_test_prompt = "Do you think it will help me stay fit?"
    prepared_prompt, context_referred = prepare_prompt(final_test_prompt)
    
    # Get the chat completion response from OpenAI
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prepared_prompt,
        max_tokens=MAX_TOKENS
    )
    
    print(f"Final Test Prompt: {final_test_prompt}")
    print(f"Context Referred: {context_referred}")
    print(f"Final Test Prompt Response: {response.choices[0].text.strip()}")

# Call the test function to generate the Final Test Prompt Response
if __name__ == "__main__":
    add_embeddings_to_pinecone(history)  # Ensure embeddings are added before testing
    test_final_prompt()
