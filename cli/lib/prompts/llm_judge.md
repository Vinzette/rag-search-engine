Rate how relevant each result is to this query on a 0-3 scale:

Query: {query}

Results:
{formatted_results}

Scale:
- 3: Highly relevant (Use this if it seems likely user would enjoy the movie based on query)
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant (Use this if it has nothing to do with query)

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]