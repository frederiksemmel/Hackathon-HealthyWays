import googlemaps
from datetime import datetime
import os
from pprint import pprint
from datetime import *
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import csv
import numpy as np
import time

from vbz_predictions import predict_marino, get_vbz_context


def dir_deptime(start, dest, dep_time):

    api = os.environ["GOOGLE_API_KEY"]
    gmaps = googlemaps.Client(key=api)
    route = gmaps.directions(
        start, dest, mode="transit", departure_time=dep_time, alternatives=True
    )

    return route


def dir_arrtime(start, dest, arr_time):

    api = os.environ["GOOGLE_API_KEY"]
    gmaps = googlemaps.Client(key=api)
    route = gmaps.directions(
        start, dest, mode="transit", arrival_time=arr_time, alternatives=True
    )

    return route


def parse_overral(route):

    distance = route.get("legs")[0].get("distance").get("value")
    duration = timedelta(seconds=route.get("legs")[0].get("duration").get("value"))

    dict = {"overall_distance": distance, "overall_duration": duration}
    return dict


def parse_steps(route):
    steps = []
    for dict in route.get("legs")[0].get("steps"):
        if dict.get("travel_mode") == "TRANSIT":

            if (
                dict.get("transit_details").get("line").get("vehicle").get("name")
                != "Tram"
                and dict.get("transit_details").get("line").get("vehicle").get("name")
                != "Bus"
            ):

                return []
            else:
                dist = dict.get("distance").get("value")
                dur = timedelta(seconds=dict.get("duration").get("value"))
                dep = dict.get("transit_details").get("departure_stop").get("name")
                dep_time = datetime.utcfromtimestamp(
                    dict.get("transit_details").get("departure_time").get("value")
                )
                arr = dict.get("transit_details").get("arrival_stop").get("name")
                arr_time = datetime.utcfromtimestamp(
                    dict.get("transit_details").get("arrival_time").get("value")
                )
                towards = dict.get("transit_details").get("headsign")
                line = dict.get("transit_details").get("line").get("short_name")
                stops = dict.get("transit_details").get("num_stops")
                type = dict.get("travel_mode")
                steps.append(
                    {
                        "type": type,
                        "dist": dist,
                        "dur": dur,
                        "dep": dep,
                        "dep_time": dep_time,
                        "arr": arr,
                        "arr_time": arr_time,
                        "line": line,
                        "towards": towards,
                        "stops": stops,
                    }
                )

        else:
            dist = dict.get("distance").get("value")
            dur = timedelta(seconds=dict.get("duration").get("value"))
            type = dict.get("travel_mode")
            instruction = dict.get("html_instructions")
            steps.append(
                {"type": type, "instruction": instruction, "dist": dist, "dur": dur}
            )

    return steps


def get_stations():
    halteList = [
        line.rstrip("\n") for line in open("../data/vbz_fahrgastzahlen/stationen.txt")
    ]  # data bus and tram stations
    return halteList


def get_caps():
    capacities = {  # capacities ToDo: get all capacities
        32: {"seats": 60, "stands": 95, "overall": 155},
        61: {"seats": 43, "stands": 54, "overall": 97},
        62: {"seats": 43, "stands": 54, "overall": 97},
        10: {"seats": 90, "stands": 130, "overall": 220},
        6: {"seats": 90, "stands": 130, "overall": 220},
        15: {"seats": 48, "stands": 72, "overall": 120},
        11: {"seats": 90, "stands": 130, "overall": 220},
    }
    return capacities


def get_directions():  # as in the VVZ data (1 or 2)
    with open(
        "../data/vbz_fahrgastzahlen/Haltestellen_Richtungen.csv", newline=""
    ) as f:  # stationen
        reader = csv.reader(f)
        endstations = list(reader)
    endstations = endstations[1:][:]

    dirs = {}  # directions mapped to endstations
    for i in range(len(endstations)):
        dirs[endstations[i][4]] = endstations[i][1]
    return dirs


def get_all_routes(start, destination, dt, timebefore):

    routes = []

    for i in range(0, timebefore, 30):

        route = dir_arrtime(
            start, destination, dt - timedelta(minutes=i)
        )  # dir_arrtime for arrivaltime; dir_deptime for deptime
        for j in range(0, len(route)):
            r = [
                parse_overral(route[j]),
                parse_steps(route[j]),
                0.0,
            ]  # [overall route infos,steps infos,rating]
            # pprint(r)
            if r not in routes:
                routes.append(r)
    return routes


def evaluate_routes(routes, capacities, dirs, halteList, vbz_context):
    for r in range(0, len(routes)):
        ratio = 0.0
        count = 0.0
        for j in range(len(routes[r][1])):

            if routes[r][1][j].get("type") == "TRANSIT":
                count += 1
                dep = process.extractOne(routes[r][1][j].get("dep"), halteList)[0]
                dep_time = routes[r][1][j].get("dep_time")
                towards = routes[r][1][j].get("towards")
                line = routes[r][1][j].get("line")
                stops = routes[r][1][j].get("stops")
                direction = dirs.get(process.extractOne(towards, halteList)[0])

                print(dep, dep_time, "Line: ", line, "towards: ", towards)

                arr = process.extractOne(routes[r][1][j].get("arr"), halteList)[0]
                arr_time = routes[r][1][j].get("arr_time")
                print(arr, arr_time)

                try:
                    cap = capacities.get(int(line)).get("overall")
                except:
                    cap = 150

                prediction = predict_marino(  # freddis prediction
                    dep,
                    dep_time,
                    arr,
                    arr_time,
                    int(stops),
                    str(line),
                    int(direction),
                    vbz_context,
                )
                ratio += prediction / cap

        if count != 0:
            ratio /= count
            routes[r][2] = ratio
            print("occupancy rate: ", ratio)
        print()
    return routes


def get_best_route(routes):
    bestratio = 1.0
    bestroute = []
    for r in range(len(routes)):  # get best route
        if routes[r][2] <= bestratio and routes[r][2] > 0.0:
            bestroute = routes[r][:]
            bestratio = routes[r][2]
    return bestroute


def prep_route_output(route):
    route[0]["overall_duration"] = str(route[0].get("overall_duration"))

    for dict in route[1]:  # for outputting datetime as string
        if dict.get("type") == "WALKING":
            dict["dur"] = str(dict.get("dur"))
        else:
            dict["arr_time"] = str(dict.get("arr_time"))
            dict["dep_time"] = str(dict.get("dep_time"))
            dict["dur"] = str(dict.get("dur"))
    return route


vbz_context = get_vbz_context()
halteList = get_stations()
capacities = get_caps()
dirs = get_directions()


finished = False
while not finished:
    try:
        start = input("From? ")
        destination = input("To? ")
        h = int(input("Hour? "))
        m = int(input("Minute? "))
        timebefore = int(input("Time flexibility in mins? "))
    except:
        print("wrong input try again")
        print()
        continue
    dt = datetime.now().replace(hour=h + 2, minute=m)  # timezone

    routes = get_all_routes(start, destination, dt, timebefore)

    print()
    print("Calculating ", len(routes), "possible routes...")
    print()
    time.sleep(2)

    routes = evaluate_routes(routes, capacities, dirs, halteList, vbz_context)

    bestroute = get_best_route(routes)

    bestroute = prep_route_output(bestroute)

    print()
    print("-----best route:-----")
    print()
    pprint(bestroute)
    print("best mean occupancy rate: ", bestroute[2])
    print()

    inp = input("do you want to start another request?y/n ")
    if inp != "y":
        finished = True
