from lib.search_utils import load_movies



def search_command(query, n_results):
    movies = load_movies()
    res=[]
    for movie in movies:
        if query.lower() in movie["title"].lower():
            res.append(movie)
        if len(res)==n_results:
            break
    return  res
