import numpy as np
import pandas as pd
import geopandas as gp
import timezonefinder
import pytz
import datetime as dt
import json
import requests

from configuration import BASE_DIR, CHECKINS, SPOTS_1, CATEGORY_STRUCTURE, TIMEZONES, BRASIL_STATES, COUNTRIES, CHECKINS_LOCAL_DATETIME, CHECKINS_7_CATEGORIES


import json
import numpy as np
import pandas as pd
import ast

def to_super_category(categories, to_super_category_dict):

    super_categories = []
    outras = []
    n = 0
    for e in categories:
        e = e.replace("[", "").replace("]", "")
        e = ast.literal_eval(e)
        category = e['name']
        try:
            category = to_super_category_dict[category]
            super_categories.append(category)
        except:
            super_categories.append(np.nan)
            outras.append(category)
            n+=1
    print("fora", n, " unicas: ", pd.Series(outras).unique().tolist())

    return np.array(super_categories)

if __name__ == "__main__":

    checkins = pd.read_csv(CHECKINS)
    print(checkins)
    pois1 = pd.read_csv(SPOTS_1)[['id', 'lat', 'lng', 'spot_categories']]
    pois1.columns = ['placeid', 'latitude', 'longitude', 'category']
    print(pois1)

    checkins = checkins.join(pois1.set_index('placeid'), on='placeid').dropna()

    with open(CATEGORY_STRUCTURE) as f:
        data = json.load(f)

    super_categories_dict = {'Other - Travel & Lodging': 'Travel', 'Urban Outfitters': 'Shopping', 'Starbucks': 'Food', 'Walmart': 'Shopping', 'Nike': 'Shopping',
                             'Gap': 'Shopping', 'Apple Store': 'Shopping', 'Best Buy': 'Shopping', 'Whole Foods': 'Shopping', 'Target': 'Shopping', 'Nightlife': 'Nightlife',
                             'T-Mobile': 'Shopping', 'In-N-Out Burger': 'Food', 'Five Guys Burgers and Fries': 'Food', 'Four Seasons': 'Travel', 'Holiday Inn': 'Travel',
                             "McDonald's": 'Food', 'Holiday Inn Express': 'Travel', 'Walgreens': 'Shopping', 'Walt Disne World Resort': 'Travel', 'Burger King': 'Food',
                             'Holiday Inn Club Vacations': 'Travel', 'Chipotle': 'Food', 'Staybridge Suites': 'Travel', 'Banana Republic': 'Shopping', 'AMC Theatre': 'Entertainment',
                             "Dukin' Donuts": 'Food', 'Other - Art & Culture': 'Entertainment', "Food & Foodies": "Food", "Cinemark Theatre": 'Entertainment', 'Other - Parks ': 'Outdoors'}
    for e in data['spot_categories']:
        super_category = e['name']
        super_url = e['url']
        inner_categories = e['spot_categories']
        for inner in inner_categories:
            inner_url = inner['url']
            inner_category = inner['name']
            inner_inner_categories = inner['spot_categories']

            for inner_inner in inner_inner_categories:
                inner_inner_url = inner_inner['url']
                inner_inner_category = inner_inner['name']

                super_categories_dict[inner_inner_category] = super_category

            super_categories_dict[inner_category] = super_category

    checkins['category'] = to_super_category(checkins['category'].tolist(), super_categories_dict)

    print(checkins)

    print("categorias únicas: ")
    print(checkins['category'].unique().tolist())

    checkins = checkins.dropna()

    #checkins.to_csv(CHECKINS_7_CATEGORIES, index_label=False, index=False)

    # {'Nike': ,
    #  'Gap',
    #  'Apple Store',
    #  'Banana Republic',
    #  'Starbucks',
    #  'Target',
    #  'AMC Theatre',
    #  'Chick-fil-A',
    #  'Conference',
    #  'Walgreens',
    #  'AT&T',
    #  'Special Event',
    #  'Brewery / Microbrewery',
    #  'Walmart',
    # 'Barnes & Noble',
    # 'Potbelly',
    # 'Nature Preserve',
    # 'Best Buy',
    # 'Other - College & Education',
    # 'IKEA',
    # 'Wedding',
    # 'Canal / Waterway',
    # 'Plaza / Square',
    # 'In-N-Out Burger',
    # 'Gate',
    # 'Five Guys Burgers and Fries',
    # 'Four Seasons',
    # 'Chipotle',
    # 'South American / Latin', 'Lake & Pond', 'Cineplex', 'Vegetarian / Vegan', 'Capitol', 'Alamo Drafthouse Cinema', 'Interactive > Lounge', 'Interactive > Panel > Design & Development', 'Tim Hortons', 'Krispy Kreme', 'Sonic', 'Other - Parks ', "McDonald's", 'Regional / State Park', "Lowe's", 'Birthday Party', 'Whataburger', 'Burger King', 'Walt Disney World Resort', 'Dairy Queen', 'Costco', 'The Home Depot', "Dunkin' Donuts", 'Other - Art & Culture', 'Big Blue Wet Thing', 'Party', 'Exhibit', 'Pool / Waterpark', 'River & Stream', 'Holiday Festivities', 'London Underground', "Trader Joe's", 'Disneyland Resort', 'Coffee Bean and Tea Leaf', 'Arts', "Victoria's Secret", 'H&M', 'FedEx Office', 'Whole Foods', 'American Apparel', "Peet's Coffee", 'Verizon', "Ben & Jerry's", 'Staybridge Suites', 'Holiday Inn Express', 'Meet Up', '24 Hour Fitness', 'Food & Foodies', 'Other - Airport', 'UPS', "Seattle's Best", 'InterContinental Hotel', 'Holiday Inn', 'Crowne Plaza', 'Cause', 'Concert', 'T-Mobile', 'LEGO Store', 'Cinemark Theatre', 'Old Navy', 'Nordstrom', 'Sports', 'National Public Lands Day', 'Protest', 'Watch Party', 'Town Hall', 'Aquatics', 'lululemon', 'REI', 'Capital Metro', 'Movie Spot', 'Life Time Fitness', 'Polling Place', 'Interactive > Party', 'Interactive > Keynote', 'Mellow Mushroom', 'World Water Day', 'Independent Event', 'Car2Go', 'Design Within Reach', 'Volunteering & Service', 'Democratic Event', 'Candlewood Suites', 'Film > Screening > Documentary Feature', 'Film > Screening > Narrative Feature', 'Film > Party', 'Interactive > Panel > Emerging...', 'Interactive > Panel > Business', 'Interactive > Core Conversation > Emerging...', 'Interactive > Panel', 'Interactive > Special Event', 'Investor', 'Interactive > Panel > Speakers Series', 'Film > Screening > Documentary Short', 'Interactive > Core Conversation > Love & Happiness', 'Interactive > Panel > Convergence', 'Film > Panel', 'Interactive > Panel > Late Break', 'Interactive > Core Conversation > Convergence', 'Interactive > Workshop', 'Interactive > Core Conversation > Greater Good', 'Film > Studio SX', 'Interactive > Panel > Greater Good', 'Interactive > Book Reading', 'Interactive > Exhibitor Events', 'Interactive > Studio SX', 'Film > Screening > Shorts Program', 'Interactive > Panel > ScreenBurn', 'Interactive > Core Conversation > ScreenBurn', 'Interactive > Book Signing', 'Music > Party', 'Taco Bueno', 'Capitol Building', 'Sprint', 'Interactive > Featured Speaker > Solo', 'Interactive > Design / Development > Solo', 'SXSW 2011', 'Interactive > Emerging > Solo', 'Interactive > Design / Development > Dual', 'Interactive > Social Graph > Panel', 'Interactive > Emerging > Dual', 'Hotel Indigo', 'bmibaby', 'RUGBY', 'Film > Comedy', 'Interactive > Featured Speaker > Keynote', 'Film > Panel > Mentor Session', 'Interactive > Featured Speaker > Panel', 'Sundance Screening', 'Interactive > Greater Good > Dual', 'Interactive > Greater Good > Core Conversation', 'Interactive > Future of Journalism > Panel', 'Interactive > Social Graph > Core Conversation', 'Interactive > Future15', 'Interactive > Greater Good > Panel', 'Interactive > Comedy', 'Interactive', 'Interactive > Business > Panel', 'Interactive > Emerging > Panel', 'Interactive > Business > Core Conversation', 'Interactive > Business > Solo', 'Interactive > ScreenBurn > Panel', 'Interactive > Branding / Marketing > Panel', 'Film > Panel > Interactive Crossover', 'Interactive > Convergence > Dual', 'Film > Panel > Case Study', 'Interactive > Design / Development > Panel', 'Film > Panel > Film Criticism', 'Interactive > Work & Happiness > Core Conversation', 'Interactive > Panel > Social Graph', 'Interactive > Late Break > Panel', 'Interactive > Bookstore Appearance', 'Interactive > Greater Good > Solo', 'Film > Panel > New Technology / Next Generation', 'Film > Panel > Business', 'Interactive > Branding / Marketing > Core Conversation', 'Interactive > ScreenBurn > Solo', 'Interactive > Podcast', 'Interactive > ScreenBurn', 'Interactive > Panel > Design / Development', 'Interactive > Solo > Late Break', 'Interactive > Late Break > Solo', 'Interactive > Social Graph > Solo', 'Social Media Week', 'Film > Panel > Workshop', 'Film > Lounge', 'Film > Panel > Contracts', 'Film > Panel > Documentary', 'Republican Event', 'Political Rally', 'Rick Perry for Governor', 'Politics', 'Food', 'Dutch Bros. Coffee', 'Hotels & Motels', 'Interactive > Featured Speaker > Dual', 'Film > Panel > Conversation', 'Interactive > Future15 > Greater Good', 'Music > Showcase > R&B', 'Charlie Crist for U.S. Senate', 'Music > Showcase > Hip-Hop/Rap', 'Music > Flatstock Performance', 'Music > SXSW Interview', 'Music > Showcase > Pop', 'Music > Showcase > Rock', 'Music > Panel > Touring/Venues', 'Music > Panel > Marketing', 'Music > Showcase > Electronic', 'Music > Showcase > Classical', 'Interactive > Tech Summit', 'Music > Trade Show', 'Interactive > Sponsored Panel > ScreenBurn', 'Interactive > Featured Speaker > Distinguished Speaker', 'Interactive > Branding / Marketing > Dual', 'LIVESTRONG Day', 'Holiday Inn Club Vacations', 'Interactive > Meet Up', 'Film > Special Event', 'Interactive > Future15 > Emerging...', 'Interactive > Convergence > Panel', 'Travel', 'Interactive > Future of Journalism > Solo', 'Owl City', 'Lotusphere2011', 'Interactive > Late Break > Dual', 'Interactive > Business > Dual', 'Interactive > ScreenBurn > Dual', 'Interactive > Daystage', 'Interactive > Solo', 'Music > Showcase > Singer-Songwriter', 'Interactive > Social Graph > Dual', 'Interactive > Special Programming > Panel', 'Music > Showcase > Funk', 'Music > Gear Alley', 'Music > Panel > Publicity', 'Music > Panel > Other', 'Music > Showcase > Metal', 'Music > Showcase > World', 'Music > Showcase > Bluegrass', 'Music > Showcase > Avant/Experimental', 'Film > Screening > Narrative Short', 'Music > Showcase > Punk', 'Film > Screening > Film Awards', 'Music > Showcase > Dance', 'Music > Showcase > Folk', 'Interactive > Trade Show', 'Interactive > Robotics > Solo', 'Film > Panel > 3D', 'Film > Screening > Music Video', 'Film > Screening > Animated Short', 'Interactive > Convergence > Solo', 'Film > Panel > Meet Up', 'Interactive > Education > Panel', 'Interactive > Emerging > Sponsored Panel', 'Film > Bookstore Appearance', 'Film > Panel > Film Festivals', 'Interactive > Solo > Self-Help / Self-Improvement', 'Music > Lounge', 'Shopping', 'Music > Panel', 'Interactive > Social Broadcasting Track > Solo', 'Music > Collectors Exhibition', 'Film > Panel > Content', 'Film', 'Music > Showcase > Reggae', 'Music > Showcase > Comedy', 'Music > Showcase > Experimental Pop', 'Music > Showcase > DJ', 'Music > Showcase > Electronic/Dance', 'Music > Showcase > Americana', 'Music > Showcase > Alt Country', 'Music > Showcase > Acoustic', 'Music > Showcase > Alternative', 'Music > Showcase > Goth', 'Music > Showcase > Country', 'Music > Showcase > Soul', 'Music > Showcase > Latin', 'Music > Showcase > Jazz', 'Music > Special Event', 'Interactive > Future15 > Convergence', 'Interactive > B2B > ScreenBurn', 'Film > Book Signing', 'Interactive > Core Conversation > Business', 'Film > Daystage', 'Music > Book Signing', 'Music > SXSW Keynote', 'Music > SXSW Featured Speaker', 'Music > Panel > History of Music', 'Interactive > Future15 > Business', 'Interactive > Screenburn', 'Film > Screening > Experimental Short', 'Music > Showcase > Latin Rock', 'Interactive > CLE > Convergence', 'Music > Showcase', 'Music > Showcase > Blues', 'Music > Panel > Branding', 'Music > Panel > Late Entry', 'Film > Panel > Production', 'Film > Panel > Marketing / PR / Publicity', 'Film > Panel > Stunts and Effects', 'Film > Panel > Casting', 'Music > Gear Alley Stage', 'Interactive > Convergence > CLE', 'Interactive > Future of Journalism > Core Conversation', 'Interactive > Future of Journalism > Dual', 'Film > Screening > Awards Program', 'Film > Trade Show', 'Interactive > Social Broadcasting Track > Panel', 'Film > Panel > Music in Film', 'Film > Panel > CLE', 'Spot Stub', 'Interactive > Yoga > Greater Good', 'Music > Panel > Media', 'Music > Dual > Online Sales/Subscriptions', 'Music > Mentor Session > Other', 'Music > Panel > Legal', 'Music > Bookstore Appearance', 'Music > Quickies > Licensing', 'Music > Demo Listening > Other', 'Music > Panel > Studio', 'Interactive > Solo > Emerging', 'Interactive > Panel > Emerging', 'Parks - Other', 'Interactive > Branding / Marketing > Solo', 'Film > Panel > Comedy', 'Interactive > ScreenBurn > Sponsored Panel', 'Interactive > Health > Solo', 'Interactive > Health > Panel', 'Film > Panel > Animation', 'Interactive > Special Programming > Solo', 'Music > SXSW Featured Speaker > Other', 'Interactive > Solo > ScreenBurn', 'Music > Quickies > Publicity', 'Film > Panel > Exhibition', 'Architecture & Buildings', 'Entertainment', 'Interactive > Social Graph > 140', 'Interactive > Social Broadcasting Track > Dual', 'Film > Panel > Distribution', 'Art & Culture - Other', 'Interactive > Latin America > Panel', 'Music > Studio SX', 'Interactive > Latin America > Solo', 'Nightlife', 'Music > Panel > International', 'Interactive > Core Conversation > Social Graph', 'Film > Panel > Funding', 'Music > Showcase > Ska', 'Interactive > Latin America > Dual', 'Interactive > Robotics > Panel', 'Interactive > 140 > Social Graph', 'Music > Comedy', 'Interactive > Greater Good', 'Interactive > Health > Dual', 'Interactive > Health > Core Conversation', 'Interactive > Panel > Social Networks', 'Interactive > Meet-Up', 'Music > SXSW Interview > Other', 'Music > Panel > Music Placement (Film/TV/Ads)', 'Interactive > Health > Meet Up', 'Music > Quickies > Music Placement', 'Music > Panel > Licensing', 'Music > Panel > A&R', 'Music > Panel > Management', 'Music > Indie Village', 'Music > Exhibitor Events', 'Interactive > Panel > Branding / Marketing / Publicity', 'Event', 'Interactive > Education > Core Conversation', 'Jim Ward for U.S. Congress', 'Music > Quickies > Management', 'Film > Screening > Texas High School', 'Music > Quickies > Studio', 'Interactive > Future15 > Education', 'Outdoors', 'Interactive > Social Graph > 140 Conference', 'Music > Showcase > Avant-garde', 'Music > Showcase > Cabaret', 'Music > Showcase > Pop ', 'Music > Showcase > Classic Rock', 'Music > SXSW Interview > History of Music', 'Music > Panel > Labels'}