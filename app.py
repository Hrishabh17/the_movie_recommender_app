import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from bs4 import BeautifulSoup
import requests
import lxml
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)

@app.route('/')
@app.route('/home')
def images():

    df=pd.read_csv('static/IMDb_Top_1000_Movies_Dataset.csv')
    random=np.random.randint(0,999, size=20)
    movie_poster_list= []
    for i in range(20):
        dict={'link':df['Movie_Poster_HD'][random[i]], 'name': df['Movie_Name'][random[i]], 'index':df['Index'][random[i]]}
        movie_poster_list.append(dict)
    return render_template('home.html', lists=movie_poster_list, random=random)

@app.route('/description/<index>')
def description(index):

    df = pd.read_csv('static/IMDb_Top_1000_Movies_Dataset.csv')
    movie_names=[]
    index=int(index)
    str=df['Movie_Genre'][index]
    str_count=str.count(',')
    genre=df['Movie_Genre'][index].split(',', maxsplit=str_count)
    len_genre=len(genre)

    cast_str=df['Movie_Cast'][index]
    cast_str_count=cast_str.count(',')
    casts=df['Movie_Cast'][index].split(',',maxsplit=cast_str_count)

    dict = {'name': df['Movie_Name'][index], 'poster': df['Movie_Poster_HD'][index], 'cast': casts,
            'description': df['Movie_Description'][index], 'rating': df['Movie_Rating'][index]
        ,'genre': genre , 'len':len_genre, 'len_cast':cast_str_count
            ,'run':df['Movie_Runtime'][index], 'certificate':df['Movie_Certificate'][index]
            ,'year':df['Movie_Year'][index]}
    movie_names.append(dict)

    file=open('static/my_model.pkl','rb')
    vectors=pickle.load(file)
    similarity=pickle.load(file)

    movie_index = df[df['Index'] == index].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:7]
    movie_poster_list=[]
    for i in movie_list:
        dict={'link':(df.iloc[i[0]].Movie_Poster_HD),'index':(df.iloc[i[0]].Index)}
        movie_poster_list.append(dict)

    return render_template('description.html', movie_name=movie_names, lists=movie_poster_list)

@app.route('/description', methods=['GET', 'POST'])
def search_description():
    df = pd.read_csv('static/IMDb_Top_1000_Movies_Dataset.csv')

    if flask.request.method =='POST':
        m_name= flask.request.form['movie_name'].lower().replace(' ','')
        z=df['Movie_Name'].apply(lambda x:x.replace(' ','')).apply(lambda y:y.lower())
        x=z.str.contains(m_name)
        # if m_name in z.values:
            # index=df[z==m_name].index[0]
        if x.any()==True:
            index=df[z.str.contains(m_name)].index[0]
            df = pd.read_csv('static/IMDb_Top_1000_Movies_Dataset.csv')
            movie_names = []
            index = int(index)
            str = df['Movie_Genre'][index]
            str_count = str.count(',')
            genre = df['Movie_Genre'][index].split(',', maxsplit=str_count)
            len_genre = len(genre)

            cast_str = df['Movie_Cast'][index]
            cast_str_count = cast_str.count(',')
            casts = df['Movie_Cast'][index].split(',', maxsplit=cast_str_count)

            dict = {'name': df['Movie_Name'][index], 'poster': df['Movie_Poster_HD'][index], 'cast': casts,
                    'description': df['Movie_Description'][index], 'rating': df['Movie_Rating'][index]
                , 'genre': genre, 'len': len_genre, 'len_cast': cast_str_count
                , 'run': df['Movie_Runtime'][index], 'certificate': df['Movie_Certificate'][index]
                , 'year': df['Movie_Year'][index]}
            movie_names.append(dict)

            cv = CountVectorizer(max_features=5000, stop_words='english')
            ps = PorterStemmer()

            def stem(text):
                y = []
                for i in text.split():
                    y.append(ps.stem(i))
                return " ".join(y)

            df['All_Movie_Info'].apply(stem)

            vectors = cv.fit_transform(df['All_Movie_Info']).toarray()
            print(vectors.shape)

            similarity = cosine_similarity(vectors)
            movie_poster_list = []

            def recommend(m_index):
                movie_index = df[df['Index'] == m_index].index[0]
                distance = similarity[movie_index]
                movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:7]

                for i in movie_list:
                    dict = {'link': (df.iloc[i[0]].Movie_Poster_HD), 'index': (df.iloc[i[0]].Index)}
                    movie_poster_list.append(dict)

            recommend(index)


            file = open('static/my_model.pkl', 'wb')
            pickle.dump(vectors, file)
            pickle.dump(similarity, file)
            file.close()


            return render_template('description.html', movie_name=movie_names, lists=movie_poster_list)

        else:
            m_name=m_name.replace(' ','+')
            m_name=m_name.lower()
            url_str=f"https://www.imdb.com/find?q={m_name}&ref_=nv_sr_sm"
            whole_html_search_list=requests.get(url_str).text
            beautify_soup=BeautifulSoup(whole_html_search_list,'lxml')
            list_movie=beautify_soup.find('table', class_='findList')
            movie_1=list_movie.find('tr', class_='findResult odd')
            movie_link=movie_1.find('a').get('href')
            imdb_link="https://www.imdb.com"
            movie_link_final=imdb_link+movie_link

            movie_page_whole_html=requests.get(movie_link_final).text
            beautify_movie_page=BeautifulSoup(movie_page_whole_html,'lxml')

            try:
                movie_title=beautify_movie_page.find('h1', class_='TitleHeader__TitleText-sc-1wu6n3d-0 dxSWFG').text
            except:
                movie_title = beautify_movie_page.find('h1', class_='TitleHeader__TitleText-sc-1wu6n3d-0 cLNRlG').text

            try:
                movie_year=beautify_movie_page.find('span', class_='TitleBlockMetaData__ListItemText-sc-12ein40-2 jedhex').text
            except:
                movie_year='NA'
            try:
                movie_certificate = beautify_movie_page.find_all('span', class_='TitleBlockMetaData__ListItemText-sc-12ein40-2 jedhex')[1].text
            except:
                movie_certificate='NA'
            movie_runtime_div = beautify_movie_page.find('ul', class_='ipc-inline-list ipc-inline-list--show-dividers TitleBlockMetaData__MetaDataList-sc-12ein40-0 dxizHm baseAlt')
            try:
                movie_runtime=movie_runtime_div.find_all('li', class_='ipc-inline-list__item')[2].text
            except:
                movie_runtime='NA'
            try:
                movie_rating=beautify_movie_page.find('span', class_='AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV').text
            except:
                movie_rating='NA'

            movie_genre=[]
            for i in range(2):
                try:
                    movie_genre.append(beautify_movie_page.find_all('span', class_='ipc-chip__text')[i].text)
                except:
                    movie_genre.append('NA')
            str_genre=''
            for i in range(len(movie_genre)):
                x=movie_genre[i]
                if (i == (len(movie_genre) - 1)):
                    str_genre = str_genre +x
                else:
                    str_genre = str_genre + x + ', '


            movie_description = beautify_movie_page.find('span', class_='GenresAndPlot__TextContainerBreakpointXL-sc-cum89p-2 eqbKRZ').text.replace('Read all', '')


            movie_cast=[]

            for i in range(3):
                try:
                    cast=beautify_movie_page.find_all('div',class_='StyledComponents__CastItemWrapper-sc-y9ygcu-7 esVIGD')[i]
                    cast_name=cast.find('a', class_='StyledComponents__ActorName-sc-y9ygcu-1 ezTgkS').text
                    movie_cast.append(cast_name)
                except:
                    movie_cast.append('NA')

            str = ''
            for i in range(len(movie_cast)):
                x= movie_cast[i]

                if (i==(len(movie_cast)-1)):
                    str=str+x
                else:
                    str=str+x+', '
            str_cast=str

            movie_poster_link=beautify_movie_page.find('a', class_='ipc-lockup-overlay ipc-focusable').get('href')
            movie_poster_link = imdb_link + movie_poster_link

            poster_html = requests.get(movie_poster_link).text
            poster_beautify = BeautifulSoup(poster_html,'lxml')


            poster_l = poster_beautify.find('div',class_='MediaViewerImagestyles__PortraitContainer-sc-1qk433p-2 iUyzNI')
            poster_link = poster_l.find_all('img')[0].get('src')


            dict={'name':movie_title, 'cast':movie_cast, 'description':movie_description, 'genre':movie_genre, 'rating':movie_rating
                  ,'run':movie_runtime,'certificate':movie_certificate,'year':movie_year,'poster':poster_link}

            all_movie_data=(movie_title + ' ' + str_cast.replace(', ', ' ') + ' ' + str_genre.replace(',',' ')+' '+movie_description.replace('.','')).lower()


            df_new=pd.DataFrame({'Index':df['Movie_Name'].shape[0],'Movie_Name':movie_title, 'Movie_Rating':movie_rating,'Movie_Certificate':movie_certificate
                                ,'Movie_Year':movie_year,'Movie_Runtime':movie_runtime,'Movie_Cast':str_cast,'Movie_Genre':str_genre,
                                 'Movie_description':movie_description ,'Movie_Poster_HD': poster_link,'All_movie_info': all_movie_data },index=[0])

            movie_name=[]
            movie_name.append(dict)

            df_new.to_csv('static/IMDb_Top_1000_Movies_Dataset.csv',mode='a',index=False, header=False)
            df_new_new=pd.read_csv('static/IMDb_Top_1000_Movies_Dataset.csv')
            print(df_new_new.shape)
            cv = CountVectorizer(max_features=5000, stop_words='english')
            ps = PorterStemmer()

            def stem(text):
                y = []
                for i in text.split():
                    y.append(ps.stem(i))
                return " ".join(y)

            df_new_new['All_Movie_Info'].apply(stem)

            vectors = cv.fit_transform(df_new_new['All_Movie_Info']).toarray()

            similarity = cosine_similarity(vectors)
            movie_poster_list = []
            def recommend(m_index):
                movie_index = df_new_new[df_new_new['Index'] == m_index].index[0]
                distance = similarity[movie_index]
                movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:7]


                for i in movie_list:
                    dict = {'link': (df_new_new.iloc[i[0]].Movie_Poster_HD), 'index': (df_new_new.iloc[i[0]].Index)}
                    movie_poster_list.append(dict)
            recommend(df.shape[0])


            file = open('static/my_model.pkl', 'wb')
            pickle.dump(vectors, file)
            pickle.dump(similarity, file)
            file.close()

            return render_template('negative.html', movie_name=movie_name, lists=movie_poster_list)


if __name__== "__main__":
    app.run(debug=True)
