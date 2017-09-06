from py_ms_cognitive import PyMsCognitiveWebSearch

search_term = "Amy Burkhardt"
search_service = PyMsCognitiveWebSearch("a185959d275247529ba4bb965f9f56ce", '"Five-star technology solutions" AND "New York City Department of Education" AND assessment')
first_result = search_service.search(limit=1, format='json')  # 1-50
print(first_result[0].title)
print(first_result[0].url)
