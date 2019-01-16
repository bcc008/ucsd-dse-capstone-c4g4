import json

def eval_and_redump(text):
    file = open('./data/'+text+'.json','r')
    write = open('./data/'+text+'_fixed.json','w')
    array = []
    for line in file:
        line_dict = eval(line)
        array.append(line_dict)
    simplejson.dump(array,write,indent='\t')
    return None

if name == '__main__':
	eval_and_redump('australian_user_reviews')
	eval_and_redump('australian_users_items')
	eval_and_redump('bundle_data')
	eval_and_redump('steam_reviews')
	eval_and_redump('steam_games')