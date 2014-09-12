"""
Self defined domain-specific stop-word list.
"""
from sklearn.feature_extraction import stop_words
from util.input import unpickle

__author__ = 'kensk8er'

# TODO: create stopword list for countries

# stopwords common for both users and jobs (derived from open source google code, + self-define)
common_stopwords = stop_words.ENGLISH_STOP_WORDS.union(
    {'ll', 've', 'a', "a's", 'able', 'about', 'above', 'abroad', 'abst', 'accordance', 'according',
     'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'adopted', 'affected', 'affecting',
     'affects', 'after', 'afterwards', 'again', 'against', 'ago', 'ah', 'ahead', "ain", 'all', 'allow',
     'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am',
     'amid', 'amidst', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any',
     'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart',
     'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'are', 'aren', "aren",
     'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'auth', 'available',
     'away', 'awfully', 'b', 'back', 'backward', 'backwards', 'be', 'became', 'because', 'become',
     'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins',
     'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond',
     'bill', 'biol', 'both', 'bottom', 'brief', 'briefly', 'but', 'by', 'c', "c'mon", "c's", 'ca', 'call',
     'came', 'can', "can't", 'cannot', 'cant', 'caption', 'cause', 'causes', 'certain', 'certainly',
     'changes', 'clearly', 'co', 'co.', 'com', 'come', 'comes', 'computer', 'con', 'concerning',
     'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding',
     'could', "couldn", 'couldnt', 'course', 'cry', 'currently', 'd', 'dare', "darent", 'date', 'de',
     'definitely', 'describe', 'described', 'despite', 'detail', 'did', "didn", 'different', 'directly',
     'do', 'does', "doesn", 'doing', "dont", 'done', 'down', 'downwards', 'due', 'during', 'e', 'each',
     'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'eleven', 'else', 'elsewhere', 'empty',
     'end', 'ending', 'enough', 'entirely', 'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'evermore',
     'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',
     'f', 'fairly', 'far', 'farther', 'few', 'fewer', 'ff', 'fifteen', 'fifth', 'fify', 'fill', 'find',
     'fire', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'forever', 'former',
     'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'from', 'front', 'full', 'further',
     'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go',
     'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn", 'half', 'happens',
     'hardly', 'has', "hasn", 'hasnt', 'have', "haven", 'having', 'he', "he'd", "he'll", "he's", 'hed',
     'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres',
     'hereupon', 'hers', 'herself', 'herse\xe2\x80\x9d', 'hes', 'hi', 'hid', 'him', 'himself',
     'himse\xe2\x80\x9d', 'his', 'hither', 'home', 'hopefully', 'how', "how's", 'howbeit', 'however',
     'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'id', 'ie', 'if', 'ignored', 'im', 'immediate',
     'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'index',
     'indicate', 'indicated', 'indicates', 'information', 'inner', 'inside', 'insofar', 'instead',
     'interest', 'into', 'invention', 'inward', 'is', "isn", 'it', "it'd", "it'll", "it's", 'itd', 'its',
     'itself', 'itse\xe2\x80\x9d', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'keys', 'kg', 'km', 'know',
     'known', 'knows', 'l', 'largely', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less',
     'lest', 'let', "let's", 'lets', 'like', 'liked', 'likely', 'likewise', 'line', 'little', 'look',
     'looking', 'looks', 'low', 'lower', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may',
     'maybe', "mayn", 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might',
     "mightn", 'mill', 'million', 'mine', 'minus', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly',
     'move', 'mr', 'mrs', 'much', 'mug', 'must', "mustn", 'my', 'myself', 'myse\xe2\x80\x9d', 'n', 'na',
     'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', "needn",
     'needs', 'neither', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'next', 'nine', 'ninety',
     'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not',
     'noted', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'o', 'obtain', 'obtained',
     'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one', "one's",
     'ones', 'only', 'onto', 'opposite', 'or', 'ord', 'other', 'others', 'otherwise', 'ought', "oughtn",
     'our', 'ours', 'ours ', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page',
     'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus',
     'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'presumably',
     'previously', 'primarily', 'probably', 'promptly', 'proud', 'provided', 'provides', 'put', 'q', 'que',
     'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably',
     'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively',
     'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'round', 'run', 's', 'said',
     'same', 'saw', 'say', 'saying', 'says', 'sec', 'second', 'secondly', 'section', 'see', 'seeing',
     'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious',
     'seriously', 'seven', 'several', 'shall', "shan", 'she', "she'd", "she'll", "she's", 'shed', 'shes',
     'should', "shouldn", 'show', 'showed', 'shown', 'showns', 'shows', 'side', 'significant',
     'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', 'slightly', 'so', 'some',
     'somebody', 'someday', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes',
     'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying',
     'state', 'states', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such',
     'sufficiently', 'suggest', 'sup', 'sure', 'system', 't', "t's", 'take', 'taken', 'taking', 'tell',
     'ten', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've",
     'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd",
     "there'll", "re", "there's", "ve", 'thereafter', 'thereby', 'thered', 'therefore',
     'therein', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'these', 'they', "they'd",
     "they'll", "they're", "they've", 'theyd', 'theyre', 'thick', 'thin', 'thing', 'things', 'think',
     'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thousand',
     'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'till', 'tip', 'to', 'together',
     'too', 'took', 'top', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twelve',
     'twenty', 'twice', 'two', 'u', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless',
     'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'upwards', 'us', 'use', 'used', 'useful',
     'usefully', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', 'versus', 'very',
     'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', "wasn", 'way', 'we', "we'd",
     "we'll", "we're", "we've", 'wed', 'welcome', 'well', 'went', 'were', "weren't", 'what', "what'll",
     "what's", "what've", 'whatever', 'whats', 'when', "when's", 'whence', 'whenever', 'where', "where's",
     'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which',
     'whichever', 'while', 'whilst', 'whim', 'whither', 'who', "who'd", "who'll", "who's", 'whod',
     'whoever', 'whole', 'whom', 'whomever', 'whos', 'whose', 'why', "why's", 'widely', 'will', 'willing',
     'wish', 'with', 'within', 'without', "win", 'wonder', 'words', 'world', 'would', "wouldn", 'www',
     'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'your', 'youre',
     'yours', 'yourself', 'yourselves', 'z', 'zero', 'hsbc', 'view', 'click', 'mailto', 'nomura', 'message', 'january',
     'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
     'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'survey', 'invitation', 'daily',
     'weekly', 'monthly', 'highlight', 'report', 'attach', 'insight', 'comment', 'morning', 'week', 'day', 'month',
     'year', 'minute', 'second', 'hour', 'two', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
     'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'comment', 'update', 'note', 'thought', 'chosen',
     'running', 'cent', 'dollar', 'ordinary', 'subscription', 'email', 'mail', 'use', 'near', 'time', 'form', 'read',
     'hundred', 'thousand', 'million', 'billion', 'trillion', 'expect', 'expectation', 'detail', 'number', 'curve',
     'end', 'member', 'offer', 'estimate', 'index', 'line', 'ensure', 'person', 'state', 'percent', 'forward', 'let',
     'thank', 'dear', 'position', 'night', 'phone', 'chart', 'target', 'land', 'document', 'attachment', 'recipient',
     'calendar', 'confirmation', 'overview', 'yesterday', 'today', 'tomorrow', 'last', 'next', 'previous', 'final',
     'summarize', 'quieter', 'whilst', 'source', 'distribution', 'headline', 'increase', 'right', 'refer', 'fall',
     'short', 'area', 'address', 'paste', 'copy', 'close', 'stop', 'start', 'sender', 'buy', 'sell', 'know', 'first',
     'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'horsey', 'alice', 'fi', 'tsy', 'cap',
     'mkts', 'james', 'coxon', 'jim', 'reid', 'john', 'normand', 'jone', 'gareth', 'joseph', 'lavorgna', 'lazzerini',
     'ettore', 'lee', 'thomas', 'lightowler', 'lin', 'li', 'linan', 'liu', 'louisa', 'lam', 'makino', 'makoto',
     'stringa', 'mark', 'wall', 'moec', 'mauro', 'spencer', 'tong', 'narminio', 'aurelien', 'nick', 'burn', 'paone',
     'fabiana', 'perry', 'kojodjojo', 'peter', 'hooper', 'torsten', 'slok', 'sidorov', 'pria', 'bakhshi', 'razzak',
     'aisha', 'robert', 'mellman', 'sgh', 'advisor', 'saliba', 'nadia', 'sameer', 'goel', 'schmieding', 'holger',
     'schulz', 'christian', 'seb', 'barker', 'sellwood', 'ross', 'shuyan', 'wu', 'small', 'cameron', 'smalley',
     'staniford', 'stephen', 'stakhiv', 'sun', 'cfa', 'tellisa', 'clarke', 'teves', 'join', 'tobieson', 'joakim',
     'tully', 'edel', 'walsh', 'paul', 'wilson', 'peter', 'wolff', 'patrick', 'wood', 'yalcin', 'ekin', 'van',
     'wagensveld', 'suzan', 'anna', 'zadornova', 'bhanu', 'baweja', 'cynthia', 'cheng', 'emily', 'benn', 'eve', 'raoul',
     'qi', 'chen', 'wang', 'tao', 'andrew', 'cate', 'angela', 'barber', 'brian', 'boucher', 'prospectus', 'edel',
     'tully', 'gustavo', 'arteta', 'gyorgy', 'kovacs', 'inigo', 'fraser', 'jenkin', 'jason', 'perl', 'normand',
     'julien', 'garran', 'larry', 'hatheway', 'loncust', 'manik', 'narain', 'martin', 'lueck', 'nelson', 'ramin',
     'nakisa', 'reinhard', 'cluse', 'reto', 'huenerwadel', 'stephane', 'deo', 'major', 'william', 'darwin', 'adam',
     'richardson', 'aditya', 'chordia', 'adrian', 'mowat', 'david', 'aserkoff', 'pedro', 'martin', 'josh', 'klaczek',
     'sanaya', 'tavaria', 'agata', 'urbanska', 'giner', 'alastair', 'newton', 'alejandro', 'martinez', 'cruz', 'alex',
     'roever', 'white', 'alexander', 'morozov', 'ander', 'mielsen', 'andre', 'carvalho', 'loes', 'de', 'silva',
     'arindam', 'sandilya', 'artem', 'biryukov', 'ben', 'laidler', 'benjamin', 'mandel', 'bert', 'lourenco', 'bertrand',
     'delgado', 'bob', 'janjuah', 'bruce', 'kasman', 'hansley', 'loey', 'lupton', 'warden', 'carsar', 'maasry',
     'charlene', 'saltzman', 'chris', 'attfield', 'clyde', 'wardle', 'constantin', 'jancso', 'crystal', 'zhao', 'fenn',
     'grosvenor', 'daragh', 'maher', 'bloom', 'faulkner', 'mackie', 'watt', 'dennis', 'debusschere', 'devendran',
     'mahendran', 'di', 'luo', 'dick', 'hokenson', 'dilip', 'shahani', 'ed', 'hyman', 'edward', 'morse', 'elise',
     'badoy', 'elizabeth', 'ernie', 'tedeschi', 'magazine', 'francis', 'khagendra', 'gupta', 'fabio', 'bassi',
     'fordham', 'tina', 'francesc', 'lleal', 'meggyesi', 'francisco', 'schumacher', 'will', 'frank', 'frederic',
     'neumann', 'fredrik', 'nerbrand', 'garry', 'evan', 'christou', 'gianluca', 'salford', 'gile', 'patternson',
     'gordian', 'kemen', 'greg', 'fuzesi', 'han', 'lorenzen', 'howard', 'wen', 'inigo', 'fraser', 'ipreo', 'ivan',
     'zubo', 'izumi', 'devalier', 'jabaz', 'mathai', 'jae', 'yang', 'jaewoo', 'nakajima', 'james', 'barne', 'pomeroy',
     'walsh', 'feroli', 'janet', 'henry', 'javier', 'finkman', 'jeanette', 'jenny', 'lai', 'jimmy', 'coonan',
     'jitendra', 'sriram', 'lomax', 'holly', 'huffman', 'zhu', 'stubbs', 'morgenstern', 'mellman', 'joyce', 'chang',
     'juan', 'carlos', 'ju', 'karen', 'ward', 'kevin', 'hebner', 'logan', 'krishana', 'guha', 'lan', 'lawrence', 'dyer',
     'leif', 'eskesen', 'lorena', 'dominguez', 'loui', 'odette', 'ma', 'xiaoping', 'malcolm', 'barr', 'protopapa',
     'marina', 'valle', 'marjorie', 'hernandez', 'mark', 'diver', 'mcdonald', 'schofield', 'mathilde', 'lemoine',
     'king', 'matthew', 'dabrowski', 'meera', 'chandan', 'melis', 'metiner', 'mufg', 'murat', 'toprak', 'ulgen',
     'murray', 'gunn', 'natixis', 'nalin', 'chutchotitham', 'nikolaos', 'panigirtzoglou', 'omfif', 'secretariat',
     'oscar', 'sloterbeck', 'charlene', 'pablo', 'pankaj', 'mataney', 'bloxham', 'mackel', 'hooper', 'sullivan', 'pin',
     'ru', 'tan', 'qu', 'hongbin', 'ramiro', 'blazquez', 'raphael', 'brun', 'aguerre', 'ratul', 'roy', 'lynch', 'parke',
     'robostox', 'ronald', 'man', 'roubini', 'ryan', 'sajjid', 'chinoy', 'sally', 'auld', 'sanaya', 'tavaria', 'mowat',
     'scott', 'chronert', 'sergio', 'shora', 'haydari', 'simon', 'well', 'stacy', 'antczak', 'cfa', 'su', 'sian', 'lim',
     'subhrajit', 'banerjee', 'junwei', 'teresa', 'cascino', 'terry', 'haine', 'theologis', 'chapsali', 'tohru',
     'sasaki', 'trinh', 'nguyen', 'le', 'vicki', 'stern', 'victor', 'fu', 'wai', 'shin', 'chan', 'wietse', 'nijenhuis',
     'wilson', 'chin', 'xavier', 'botteri', 'yi', 'hu', 'zhi', 'ming', 'zhang', 'brought', 'here', 'about', 'use',
     'confuse', 'uk', 'united-kingdom', 'sex', 'gender'})

# stopwords for person's names
name_stopwords = unpickle('data/others/names.pkl')

# stopwords for school names
# school_stopwords = unpickle('data/others/schools.pkl')
school_stopwords = set()  # FIXME: school stopwords are not good enough (lots of noise) so excluded for now

# stopwords for school names
location_stopwords = unpickle('data/others/locations.pkl')

# stopwords for school names
city_stopwords = unpickle('data/others/cities.pkl')

# stopwords specific for users
user_stopwords = common_stopwords.union(name_stopwords).union(school_stopwords).union(location_stopwords).union(
    city_stopwords).union(
    {'test'})

# stopwords specific for jobs
job_stopwords = common_stopwords.union(name_stopwords).union(school_stopwords).union(location_stopwords).union(
    city_stopwords).union(
    {'nbsp', 'font', 'family', 'arial' 'helvetica', 'san', 'serif', 'color', 'verdana', 'autospace', 'ul', 'li', 'br',
     'strong', 'table', 'center', 'dd', 'dt', 'dl', 'div', 'ol', 'span', 'td', 'tr', 'th'})

wiki_stopwords = common_stopwords.union(name_stopwords).union(school_stopwords).union(location_stopwords).union(
    city_stopwords).union(
    {'test'})
