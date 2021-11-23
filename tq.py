from fuzzywuzzy import fuzz
from google.cloud import datastore, storage, vision
import datetime
import random
import string
import numpy as np
from numpy import ndarray
import os
import cv2
from copy import deepcopy
from PIL import Image, ImageDraw
import re
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
import time
from itertools import groupby, count

datastore_client = datastore.Client()


class FuzzyComparator(object):
    def __init__(self, value, threshold=1):
        self.value = value
        self.threshold = threshold

    def __str__(self):
        return str(self.value)

    def __contains__(self, query):
        # Handle warnings thrown by FuzzyWuzzy for symbols
        if len(query) == 1 and not str(query).isalpha():
            return False
        for item in self.value:
            if (
                fuzz.token_set_ratio(query, item)
                >= min(85, ((len(query) - self.threshold) / len(query)) * 100)
                and abs(len(query) - len(item)) <= self.threshold
            ):
                return True
        return False

    def __eq__(self, other):
        for index, item in enumerate(other):
            # Handle warnings thrown by FuzzyWuzzy for symbols
            if len(item) == 1 and not str(item).isalpha():
                return False
            if (
                fuzz.partial_ratio(item, self.value[index])
                >= min(
                    85,
                    ((len(item) - 1) / len(item)) * 100,
                )
                and not abs(len(item) - len(self.value[index])) <= 1
            ):
                return False
            elif not fuzz.partial_ratio(item, self.value[index]) >= min(
                85, ((
                    len(item) - 1) / len(item)) * 100
            ):
                return False
        return True

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return self.value[index]


def generate_sentences(document, lowers, shape, min_x=0, min_y=0, max_x=9999, max_y=9999, ignore_words=[]):
    words = []
    centers = []
    for doc in document:
        if doc['description'] in FuzzyComparator(ignore_words):
            continue
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc['description'])
        centers.append([xc, yc])

    sentences = {}
    lowers.append(shape)
    lowers.sort()

    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] <= y and min_x < value[0] < max_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
            else:
                if lowers[i - 1] <= value[1] <= y and max_x > value[0] > min_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
        z = [x for _, x in sorted(zip(center_x, s))]
        sentences[str(y)] = z

    final_sentences = []
    for value in sentences.values():
        sent = ''
        for i, v in enumerate(value):
            try:
                if v.isalnum() and value[i + 1].isalnum():
                    sent += v + ' '
                elif v.isalnum() and not value[i + 1].isalnum():
                    sent += v
                elif not v.isalnum() and value[i + 1].isalpha():
                    sent += v
                elif not v.isalnum() and value[i + 1].isdigit():
                    if v == '.' or v == '-':
                        sent += v
                    else:
                        sent += v + ' '
                elif not v.isalnum() and not value[i + 1].isalnum():
                    sent += v + ' '
            except IndexError:
                sent += v
        sent = sent.replace('\n', ' ')
        if len(sent) < 4:
            continue
        final_sentences.append(sent)

    return final_sentences


def dict_sentences(document, lowers, shape, min_x=0.0, min_y=0.0, max_x=9999, max_y=9999):
    words = []
    centers = []
    for doc in document:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc)
        centers.append([xc, yc])
    sentences = {}

    lowers.append(shape)
    lowers.sort()

    all_words = []
    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] <= y and min_x < value[0] < max_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
                    all_words.append(key)
            else:
                if value[1] >= lowers[i - 1] and value[1] <= y and max_x > value[0] > min_x and max_y > value[
                    1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
                    all_words.append(key)

        center_x = fix_duplicates(center_x)
        z = [x for _, x in sorted(zip(center_x, s))]

        sentences[str(y)] = z
    return sentences, all_words


def fetch_from_datastore(search_key):
    try:
        key = datastore_client.key('bank_details', search_key)
        result = datastore_client.get(key)
        micr_code = result.get('micr_code')
        bank_name = result.get('bank_name')
        branch = result.get('bank_branch')
    except Exception as e:
        return ["", "", ""]
    return [bank_name, branch, micr_code]


def generate_random_file_name(length=20, extension=None):
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    salt = "".join(
        random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase)
        for _ in range(length - len(timestamp))
    )
    if extension:
        return "{}{}.{}".format(timestamp, salt, extension).replace("..", ".")
    return "{}{}".format(timestamp, salt)


def upload_file_to_google_cloud(filename, content_type='image/jpeg'):
    storage_client = storage.Client()

    # FOR LABS
    # bucket = storage_client.get_bucket("labs-yolo-prediction-bucket")

    # FOR PRODUCTION
    bucket = storage_client.get_bucket("yolo_model_prediction_bucket")

    # FOR IPRU
    # bucket = storage_client.get_bucket("dolphin-ekyc-api-v2")

    blob = bucket.blob(filename)
    blob.upload_from_filename(filename, content_type=content_type)
    blob.make_public()

    return blob.public_url


def store_image_on_google_drive(cropped_image, content_type='image/jpeg', format_='JPEG'):
    if not isinstance(cropped_image, ndarray):
        return None
    name = "/tmp/" + generate_random_file_name(extension=format_.lower())
    cv2.imwrite(name, cropped_image)
    cropped_image_google_bucket_url = upload_file_to_google_cloud(filename=name, content_type=content_type)
    os.remove(name)

    return cropped_image_google_bucket_url


def fix_duplicates(mylist):
    newlist = []
    for i, v in enumerate(mylist):
        totalcount = mylist.count(v)
        count = mylist[:i].count(v)
        newlist.append(v + (count * 0.1) if totalcount > 1 else v)
    return newlist


def text_within(document, x1, y1, x2, y2, height='', points=0):
    text = ""
    three_data_point_coordinates = ""
    if height:
        step = height/6
        three_data_point_coordinates = [round(y1+(x*step)) for x in range(1, points*2, 2)]

    word_centers = []
    for page in document['pages']:
        for block in page['blocks']:
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    temp_txt = ""
                    word_inside = False
                    symbol_centers = []
                    for symbol in word['symbols']:
                        try:
                            min_x = min(symbol['bounding_box']['vertices'][0]['x'], symbol['bounding_box']['vertices'][1]['x'],
                                        symbol['bounding_box']['vertices'][2]['x'], symbol['bounding_box']['vertices'][3]['x'])
                            max_x = max(symbol['bounding_box']['vertices'][0]['x'], symbol['bounding_box']['vertices'][1]['x'],
                                        symbol['bounding_box']['vertices'][2]['x'], symbol['bounding_box']['vertices'][3]['x'])
                            min_y = min(symbol['bounding_box']['vertices'][0]['y'], symbol['bounding_box']['vertices'][1]['y'],
                                        symbol['bounding_box']['vertices'][2]['y'], symbol['bounding_box']['vertices'][3]['y'])
                            max_y = max(symbol['bounding_box']['vertices'][0]['y'], symbol['bounding_box']['vertices'][1]['y'],
                                        symbol['bounding_box']['vertices'][2]['y'], symbol['bounding_box']['vertices'][3]['y'])
                            if x1 < (min_x+max_x)/2 < x2 and y1 < (min_y+max_y)/2 < y2:  # min_x >= x1 and max_x <= x2 and min_y >= y1 and max_y <= y2:
                                try:
                                    text += symbol['text']
                                    temp_txt += symbol['text']
                                    if height:
                                        symbol_centers.append((min_y+max_y)/2)
                                        word_inside = True
                                    if symbol['property']['detected_break']['type'] == 'SPACE' or symbol['property']['detected_break']['type'] == 'EOL_SURE_SPACE':
                                        text += ' '
                                    if symbol['property']['detected_break']['type'] == 'SURE_SPACE':
                                        text += '\t'
                                    if symbol['property']['detected_break']['type'] == 'LINE_BREAK':
                                        text += '\n'
                                except KeyError:
                                    continue
                        except KeyError:
                            continue
                    if word_inside:
                        word_centers.append([temp_txt, np.mean(symbol_centers)])
    return text, [word_centers, three_data_point_coordinates]


def generate_final_sentence(sentences):
    final_sentences = []
    for value in sentences.values():
        sent = ''
        for i, v in enumerate(value):
            try:
                if (v['description'].isalnum() or bool(re.match(r'\d+.\d+', v['description']))) and (value[i + 1]['description'].isalnum() or bool(re.match(r'\d+.\d+', value[i + 1]['description']))):
                    sent += v['description'] + ' '
                elif (v['description'].isalnum() or bool(re.match(r'\d+.\d+', v['description']))) and not value[i + 1]['description'].isalnum():
                    sent += v['description']
                elif not v['description'].isalnum() and value[i + 1]['description'].isalpha():
                    sent += v['description']
                elif not v['description'].isalnum() and value[i + 1]['description'].isdigit():
                    if v['description'] == '.' or v['description'] == '-':
                        sent += v['description']
                    else:
                        sent += v['description'] + ' '
            except IndexError:
                sent += v['description']
        sent = sent.replace('\n', ' ')
        if len(sent) < 4:
            continue
        final_sentences.append(sent)
    return final_sentences


def custom_generate_sentences(document, lowers, shape, min_x=0.0, min_y=0.0, max_x=9999, max_y=9999):
    words = []
    centers = []
    for doc in document:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4
        words.append(doc)
        centers.append([xc, yc])
    sentences = {}
    i = 0
    lowers.append(shape)
    lowers.sort()
    all_words = []
    for i, y in enumerate(lowers):
        s = []
        center_x = []
        for key, value in zip(words, centers):
            if i == 0:
                if value[1] <= y and min_x < value[0] < max_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
                    all_words.append(key)
            else:
                if lowers[i - 1] <= value[1] <= y and max_x > value[0] > min_x and max_y > value[1] > min_y:
                    s.append(key)
                    center_x.append(value[0])
                    all_words.append(key)

        center_x = fix_duplicates(center_x)
        z = [x for _, x in sorted(zip(center_x, s))]
        sentences[str(y)] = z
    return sentences, all_words


def get_location_for_sentence(sentences, to_find):
    word_location = False
    for sentence in sentences:
        sent = ' '.join([x['description'] for x in sentences[sentence]])
        sent = sent.lower()

        if to_find in sent:
            ind = sent.split().index(to_find.split()[0])
            first_word = sentences[sentence][ind]
            last_word = sentences[sentence][ind + len(to_find.split()) - 1]

            word_location = deepcopy(first_word)
            word_location['description'] = to_find
            word_location['bounding_poly']['vertices'][1]['x'] = last_word['bounding_poly']['vertices'][1]['x']
            word_location['bounding_poly']['vertices'][1]['y'] = last_word['bounding_poly']['vertices'][1]['y']

            word_location['bounding_poly']['vertices'][2]['x'] = last_word['bounding_poly']['vertices'][2]['x']
            word_location['bounding_poly']['vertices'][2]['y'] = last_word['bounding_poly']['vertices'][2]['y']
            return word_location


def check_box_height(word, flag, height_threshold):
    if not flag:
        return False
    y_vertices = [x1['y'] for x1 in word['bounding_poly']['vertices']]
    if (((y_vertices[3] - y_vertices[0]) + (y_vertices[2] - y_vertices[1])) / 2) < height_threshold:
        return True
    return False


def get_straight_line_coordinates(document, image_shape, reduction_ratio=0.3, height_threshold=40):
    new_image = Image.new('1', (int(image_shape[1]), int(image_shape[0])))
    draw = ImageDraw.Draw(new_image)

    for i, doc in enumerate(document):
        if check_box_height(doc, False, height_threshold):
            continue
        pil_input = []
        xs = [x1['x'] for x1 in doc['bounding_poly']['vertices']]
        ys = [x1['y'] for x1 in doc['bounding_poly']['vertices']]
        ht = ((ys[3] - ys[0]) + (ys[2] - ys[1])) / 2

        ys[0] = ys[0] + ht * reduction_ratio
        ys[1] = ys[1] + ht * reduction_ratio
        ys[2] = ys[2] - ht * reduction_ratio
        ys[3] = ys[3] - ht * reduction_ratio

        for j in range(4):
            pil_input.append(xs[j])
            pil_input.append(ys[j])
        draw.polygon(pil_input, fill=1, outline=1)

    pil_image = new_image.convert('RGB')
    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.reduce(gray, 1, cv2.REDUCE_AVG).reshape(-1)
    th = 2
    H, W = img.shape[:2]
    lowers = [y for y in range(H - 1) if hist[y + 1] <= th < hist[y]]

    for y in lowers:
       img=cv2.line(img, (0, y), (W, y), (0, 255, 0), 1)
    # cv2.imwrite("/Users/monarkunadkat/Desktop/output_folder/horizontal_seperations.png", img)

    return lowers


def get_words_within_bounds(vision_response, x1, y1, x2, y2):
    updated_document = []
    for words in vision_response['text_annotations'][1:]:
        xc = sum([x1['x'] for x1 in words['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in words['bounding_poly']['vertices']]) / 4
        if x1 < xc < x2 and y1 < yc < y2:
            updated_document.append(words)

    return updated_document


def extract_form16_table(x1, y1, x2, y2, vision_response, image_shape):
    quarter_lst = []
    updated_document = get_words_within_bounds(vision_response, x1, y1, x2, y2)
    lowers = get_straight_line_coordinates(updated_document, image_shape)
    sentence_dict, all_words = custom_generate_sentences(updated_document, lowers, image_shape[0])
    upper = get_location_for_sentence(sentence_dict, 'section 200')
    format2 = False
    if not upper:
        upper = get_location_for_sentence(sentence_dict, "of section")#'quarterly statements')
        format2 = True
        if not upper:
            return []


    sentence_dict, all_words = custom_generate_sentences(updated_document, lowers, image_shape[0], min_y=upper['bounding_poly']['vertices'][2]['y'])
    table = generate_final_sentence(sentence_dict)

    if len(table) >= 5:
        for i, row in enumerate(table[:-1]):
            if 'Total' in row:
                break

            quarter_dict = dict()
            columns = row.replace('|',' ').split()

            if len(columns) < 4:
                continue

            if not re.findall(r'[A-Z]{8}', row):
                columns.insert(1, "")

            quarter_dict['quarter'] = "Q"+str(i+1)
            quarter_dict['quarter_receipt_no'] = columns[-4]
            quarter_dict['quarter_amount_credited'] = columns[-3]
            quarter_dict['quarter_amount_deducted'] = columns[-2]
            quarter_dict['quarter_amount_remitted'] = columns[-1]
            quarter_lst.append(quarter_dict)

    else:
        # for i, row in enumerate(table[1 if format2 else 0:-1 if not format2 else len(table)]): # quarterly
        for i, row in enumerate(table[:-1 if not format2 else len(table)]):
            if 'Total' in row:
                break

            quarter_dict = dict()
            columns = row.replace('|',' ').split()

            if len(columns) < 4:
                continue

            if not re.findall(r'[A-Z]{8}', row):
                columns.insert(1, "")

            if not format2 :
                quarter_dict['quarter'] = columns[0] if len(columns) == 5 else ""
                quarter_dict['quarter_receipt_no'] = columns[-4]
                quarter_dict['quarter_amount_credited'] = columns[-3]
                quarter_dict['quarter_amount_deducted'] = columns[-2]
                quarter_dict['quarter_amount_remitted'] = columns[-1]
                quarter_lst.append(quarter_dict)

            else:
                quarter_dict['quarter'] = columns[0] if len(columns) == 4 else ""
                quarter_dict['quarter_receipt_no'] = columns[-3]
                quarter_dict['quarter_amount_credited'] = ""
                quarter_dict['quarter_amount_deducted'] = columns[-2]
                quarter_dict['quarter_amount_remitted'] = columns[-1]
                quarter_lst.append(quarter_dict)

    return quarter_lst


def check_date(date, ack, sent):
    match = re.search(r'\d{2}-\d{2}-\d{4}', date['value'] if date['value'] else sent)
    if  match: #False
        date['value'] = match.group()
    else:
        try:
            value = datetime.datetime.strptime(ack['value'][-6:], '%d%m%y').strftime('%d-%m-%Y') if ack['value'] else ""
            confidence = ack['confidence']
        except ValueError:
            value = ""
            confidence = ""
        date = {'value': value, 'confidence': confidence}
    return date


# FOR TABLE PARSING FROM US BANK
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def as_range(iterable):
    l = list(iterable)
    return int(np.average(l))


def get_columns_coordinates(src):
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    vertical = np.copy(bw)

    rows = vertical.shape[0]
    verticalsize = int(rows / 30)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # cv2.imwrite("/Users/monarkunadkat/Desktop/img_vertical8.png", vertical)

    minLineLength = 100
    maxLineGap = 200
    lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    lines = np.array(lines)

    try:
        xs = lines[:, :, 0][:]
        xs = list(set(xs[:, 0]))
        xs.sort()
        return [as_range(g) for _, g in groupby(xs, key=lambda n, c=count(): n - next(c))]
    except:
        return False


def remove_rows_from_table(image):
    img = image
    # cv2.imwrite('/Users/monarkunadkat/Desktop/fffffff.png', image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]  #

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(c)
        if x == 0 or y == 0:
            img[y - 2:y + h + 2, x:x + w + 1] = 255  # Taking two pixels above and below and making them white too
        else:
            # If the bounding box doesn't start from zeroth pixel then we can use a pixel before the x co-od also
            img[y - 2:y + h + 2, x - 1:x + w + 1] = 255

    # cv2.imwrite('/Users/monarkunadkat/Desktop/output_folder/rowremoved_{}.jpg'.format(str(time.time())), img)

    return img


def detect_column_seperations_using_sliding_window(img, original_image, vision_response, x_start, y_start, y_stop):
    h, w = img.shape
    window_size = int(w * 0.35)

    inv_img = 255 - img
    dilation_kernel = np.ones((27, 6), np.uint8)  # Horizontal kernel . (35,2) itr = 10 (2,10) 3  (25,4) 2
    dilation = cv2.dilate(inv_img, dilation_kernel, iterations=2)

    inv_dilation = cv2.bitwise_not(dilation)

    erosion_kernel = np.ones((2, 6), np.uint8)  # Horizontal kernel .  (1,15) itr = 3 (1,5) 3  (2,6) 3
    erosion = cv2.erode(dilation, erosion_kernel, iterations=3)

    # Taking column-wise sum of pixel values in image (returns a list)
    img_col_sum = np.sum(erosion, axis=0).tolist()

    # Normalising the values of img col sum
    for i in range(len(img_col_sum)):
        img_col_sum[i] = img_col_sum[i] / max(img_col_sum)

    # Taking a window based approach
    # img_col_sum=[sum(img_col_sum[i:i+100]) for i in range(len(img_col_sum)-100)]
    deviation = 300 if window_size > 300 else window_size

    peaks = []
    for i in range(0, len(img_col_sum), window_size):
        i = 0 if i == 0 else (i - deviation)
        # i = 0 if i == 0 else (i - 300)  # version2

        window_val = img_col_sum[i:i + window_size + 300]

        ysmoothed_15 = gaussian_filter1d(window_val, sigma=15)

        # Getting minimas of the smoothened graph and plotting
        min_peaks_15, _ = find_peaks(-1 * ysmoothed_15)
        a = np.array(ysmoothed_15)

        # max_minima_val = 0
        max_minima_val = np.min(ysmoothed_15[min_peaks_15] if len(min_peaks_15) != 0 else 0)

        # Getting maximas of the smoothened graph and plotting
        max_peaks_15, _ = find_peaks(ysmoothed_15)
        a = np.array(ysmoothed_15)

        max_maxima_val = np.max(ysmoothed_15[max_peaks_15] if len(max_peaks_15) != 0 else 1)

        # ---------------------------- calculate platue ----------------------------?
        # -------- first check if platue region is below threshold then only draw column ----------?
        diff_pl = np.diff(window_val)
        gradient = np.sign(diff_pl)
        if gradient[0] == 0:
            peaks.append(i)

        diff = max_maxima_val - max_minima_val
        # if difference between minima and maxima is greater than certain threshold then only there is variation which can be considered as column

        if diff > 0.1:
            height = max_minima_val + diff / 1.5
            # if window_val[0] < height:
            #     diff = np.diff(window_val)
            #     gradient = np.sign(diff)
            #     if gradient[0] == 0:
            #         peaks.append(i)

            ysmoothed_20 = gaussian_filter1d(window_val, sigma=15)

            win_peaks_20, _ = find_peaks(-1 * ysmoothed_20, height=-1 * height)
            a = np.array(ysmoothed_20)
            x_cord_win_peaks = [x + i for x in win_peaks_20]
            peaks.extend(x_cord_win_peaks)

    # cv2.imwrite('/Users/monarkunadkat/Desktop/original_image.jpeg', original_image)

    peaks = np.array(sorted(list(set(peaks)))) + x_start

    th_no_contours_crossed = 0
    th_no_contours_touched = 0
    lst = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    x3 = []
    y3 = []
    x4 = []
    y4 = []
    v = []
    # lower_x = []
    # lower_y = []
    # higher_x = []
    # higher_y = []
    x = []
    y = []
    center = []

    des = []
    for doc in vision_response:
        v = doc['bounding_poly']['vertices']
        # v = doc.bounding_poly.vertices
        _x1 = v[0]['x']
        _y1 = v[0]['y']

        _x2 = v[1]['x']
        _y2 = v[1]['y']

        _x3 = v[2]['x']
        _y3 = v[2]['y']

        _x4 = v[3]['x']
        _y4 = v[3]['y']

        _lower_x = np.min([_x1, _x2, _x3, _x4])
        _higher_x = np.max([_x1, _x2, _x3, _x4])

        _lower_y = np.min([_y1, _y2, _y3, _y4])
        _higher_y = np.max([_y1, _y2, _y3, _y4])

        x1.append(v[0]['x'])
        y1.append(v[0]['y'])

        x2.append(v[1]['x'])
        y2.append(v[1]['y'])

        x3.append(v[2]['x'])
        y3.append(v[2]['y'])

        x4.append(v[3]['x'])
        y4.append(v[3]['y'])

        x.append([_lower_x, _higher_x])

        y.append([_lower_y, _higher_y])

        x_mid = int(np.abs((_x4 - _x1)) / 2)
        center.append(_x1 + x_mid)

        des.append([doc['description'], [_lower_x, _higher_x], [_lower_y, _higher_y]])

    s = sorted(x)

    # Remove crossing peaks
    crossing_peaks = []
    for p in peaks:
        th_no_contours_crossed = 0
        for sublist in s:
            if (sublist[1] >= (p + 7) >= sublist[0]) or (sublist[1] >= (p) >= sublist[0]) or (
                sublist[1] >= (p - 1) >= sublist[0]):
                th_no_contours_crossed = th_no_contours_crossed + 1
                if th_no_contours_crossed >= 1:
                    crossing_peaks.append(p)
                    break

    final_peaks = [x for x in peaks if x not in crossing_peaks]

    # closertodata_peaks = []
    # for p in peaks:
    #     for mid in center:
    #         distance = np.abs((p - mid))

    #         if distance < 10:
    #             closertodata_peaks.append(p)

    # closertodata_peaks = list(set(closertodata_peaks))

    # final_peaks.extend(closertodata_peaks)

    temp = sorted(list(set(final_peaks)))
    ###################################################  no of words between 2 peaks
    peaks_to_be_removed = []
    no_of_words_between_cols = 2

    for i in range(len(temp) - 1):
        l_ = list(mid for mid in center if temp[i] < mid < temp[i + 1])
        if 0 < len(l_) <= no_of_words_between_cols:
            peaks_to_be_removed.append(temp[i])

    peaks_after = set(temp) - set(peaks_to_be_removed)

    final_peaks = list(sorted(peaks_after))

    # Remove extra empty peaks
    temp = sorted(list(set(final_peaks)))

    mearged_peaks = [1] * len(temp)
    after_mearge = []

    for i in range(len(temp) - 1):
        l = list(mid for mid in center if temp[i] < mid < temp[i + 1])
        if len(l) == 0:
            new_peak = int((temp[i] + temp[i + 1]) / 2)
            after_mearge.append(new_peak)
            mearged_peaks[i] = 0

    ranges = zero_runs(mearged_peaks)

    for r in ranges:
        merged_peaks = temp[r[0]:r[1] + 1]
        final_peaks = [x for x in final_peaks if x not in merged_peaks]
        merged = int(np.mean(merged_peaks))
        final_peaks.append(merged)

    for i in final_peaks:
        final_img = inv_img
        _i = (w - 1) if (i + 4) > w else (i + 4)
        inv_img[:, _i] = 255
        original_image[y_start:y_stop, i + 4] = 0

    # cv2.imwrite('/Users/monarkunadkat/Desktop/cropped.jpeg', inv_img)
    # cv2.imwrite('/Users/monarkunadkat/Desktop/{}_original_image.jpeg'.format(str(time.time())), original_image)

    return final_peaks


def get_between(document, min_x=0.0, min_y=0.0, max_x=9999, max_y=9999):
    words = []
    centers = []
    for doc in document:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4

        if min_x < xc < max_x and min_y < yc < max_y:
            words.append(doc['description'])
            centers.append(xc)
    z = [x for _, x in sorted(zip(centers, words))]
    return z


def update_vision_response(vision_response, x1, x2, y1, y2):
    updated_response = []
    for doc in vision_response['text_annotations'][1:]:
        xc = sum([x1['x'] for x1 in doc['bounding_poly']['vertices']]) / 4
        yc = sum([x1['y'] for x1 in doc['bounding_poly']['vertices']]) / 4

        if x1 < xc < x2 and y1 < yc < y2:
            updated_response.append(doc)
    return updated_response


def remove_noise(image, updated_response, x1, x2, y1, y2):
    height = int(np.abs(y2 - y1))
    width = int(np.abs(x2 - x1))

    image = image[y1:y2, x1:x2]
    new_img = np.full((height, width, 3), 255, np.uint8)

    for i, doc in enumerate(updated_response):
        xs = [x1['x'] for x1 in doc['bounding_poly']['vertices']]
        ys = [x1['y'] for x1 in doc['bounding_poly']['vertices']]
        new_img[ys[0]-y1:ys[2]-y1, xs[0]-x1:xs[1]-x1] = image[ys[0]-y1:ys[2]-y1, xs[0]-x1:xs[1]-x1]

    # cv2.imwrite('/Users/monarkunadkat/Desktop/onlytext_img_{}.jpg'.format(str(time.time())), new_img)

    return new_img  # remove_rows_from_table(new_img[y1_modified:y2_modified, x1:x2], filename)

def extract_table_data_from_us_bs(vision_response, image, x1, x2, y1, y2, filename, bank_statement_type,area_type,doc_class):

    y1_modified = int(y1+((y2-y1)*0.27))
    y2_modified = int(y2-((y2-y1)*0.2))

    x1 = 0 if x1 < 0 else x1

    reduction_ratio = 1.5

    updated_response = update_vision_response(vision_response, x1, x2, y1_modified, y2_modified)

    #print('modified image coordinates')
    #print(y1_modified,y2, x1,x2)

    height = int(np.abs(y2-y1_modified))
    width = int(np.abs(x2-x1))

    #print('image shape ============= ', image.shape)

    new_img = np.full(image.shape,255,np.uint8)

    for i, doc in enumerate(updated_response):
        pil_input = []
        xs = [x1['x'] for x1 in doc['bounding_poly']['vertices']]
        ys = [x1['y'] for x1 in doc['bounding_poly']['vertices']]

        #ht = ((ys[3] - ys[0]) + (ys[2] - ys[1])) / 2
        width = ((xs[3] - xs[2]) + (xs[1] - xs[0])) / 2

        xs[0] = int(xs[0] + width * reduction_ratio)
        xs[1] = int(xs[1] - width * reduction_ratio)
        xs[2] = int(xs[2] + width * reduction_ratio)
        xs[3] = int(xs[3] - width * reduction_ratio)

        #new_img[ys[0]-y1:ys[2]-y1, xs[0]-x1:xs[1]-x1] = image[ys[0]-y1:ys[2]-y1, xs[0]-x1:xs[1]-x1]
        new_img[ys[0]:ys[2],xs[0]:xs[1]] = image[ys[0]:ys[2],xs[0]:xs[1]]

    #cv2.imwrite('/Users/foram.jivani/Desktop/USA Testing/onlytext_img_{}.jpg'.format(str(time.time())), new_img[y1_modified:y2_modified, x1:x2])

    #step1_remove_rows = remove_noise(image, updated_response, x1, x2, y1_modified, y2_modified)
    step1_remove_rows = new_img[y1_modified:y2_modified, x1:x2] # remove_rows_from_table(new_img[y1_modified:y2_modified, x1:x2], filename)

    #import pdb; pdb.set_trace()
    column_seperations = detect_column_seperations_using_sliding_window(cv2.cvtColor(step1_remove_rows, cv2.COLOR_BGR2GRAY), image, updated_response, x1, y1, y2)
    # column_seperations = np.array(column_seperations)+x1

    updated_response = update_vision_response(vision_response, x1, x2, y1, y2)
    lowers_ = get_straight_line_coordinates(updated_response, image.shape)
    sentence_dict, all_words = dict_sentences(updated_response, lowers_, image.shape[0], min_x=x1, min_y=y1, max_x=x2, max_y=y2)

    final_sentences = generate_final_sentence(sentence_dict)
    #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
    #for sentence in final_sentences:
    #    print(sentence.encode('utf8'))

    column_seperations.append(x2)

    column_seperations = sorted(column_seperations)


    data_array = [[''] * len(column_seperations[1:]) for x in lowers_]

    lowers_.insert(0, 0)
    for i, hs in enumerate(lowers_):
        for j, vs in enumerate(column_seperations):
            try:
                z = get_between(all_words, min_x=vs, max_x=column_seperations[j + 1], min_y=hs,
                                max_y=lowers_[i + 1])
                data_array[i][j] = ' '.join(z)
            except IndexError:
                pass
    bank_with_columns = {
                          "keys": [
                                {
                                "bank_name" : "bank of america",
                                "table_type" : "transaction",
                                "columns" : ["Date", "Description","Amount"]
                                }
                            ]
                        }

    monthsShort="Jan|Feb"
    monthsLong="January|February"
    months="(" + monthsShort + "|" + monthsLong + "|" + "[\d]{0,2}" +")"
    separators = "[-/']?"
    days = r"\d{0,2}"
    years = r"\d{0,4}"

    regex1 = "^"+months + separators + days + separators + years +"$"
    regex2 = "^" + days + separators + months + separators + years +"$"

    date_regex = regex2#[regex1, regex2]

    amount_regex = r"[-]?[$]?[(]?\d+([., ]\d*)+[)]?"

    description_regex=r"[a-zA-Z]+"

    number_regex = r"^([#\d#]+)$"

    ##################  step 1 check for headers are exists or not in detected table area
    # check if line contains both date and amount then it is header in that table area

    columns = []
    table_header_found = False

    if any(x in " ".join(data_array[0]).lower().split() for x in ['date','paid']) and any(x in " ".join(data_array[0]).lower().split() for x in ['amount', 'credit', 'balance', 'debit', 'additions', 'subtractions']):
        columns = [x.lower() for x in data_array[0]]
        rows = data_array[1:]
        table_header_found = True
    elif any(x in " ".join(data_array[1]).lower().split() for x in ['date','paid']) and any(x in " ".join(data_array[1]).lower().split() for x in ['amount', 'credit', 'balance', 'debit', 'additions', 'subtractions']):
        columns = [x.lower() for x in data_array[1]]
        rows = data_array[2:]
        table_header_found = True
    elif any(x in " ".join(data_array[2]).lower().split() for x in ['date','paid']) and any(x in " ".join(data_array[2]).lower().split() for x in ['amount', 'credit', 'balance', 'debit', 'additions', 'subtractions']):
        columns = [x.lower() for x in data_array[2]]
        rows = data_array[3:]
        table_header_found = True
    elif any(x in " ".join(data_array[3]).lower().split() for x in ['date','paid']) and any(x in " ".join(data_array[3]).lower().split() for x in ['amount', 'credit', 'balance', 'debit', 'additions', 'subtractions']):
        columns = [x.lower() for x in data_array[3]]
        rows = data_array[4:]
        table_header_found = True
    elif any(x in " ".join(data_array[4]).lower().split() for x in ['date','paid']) and any(x in " ".join(data_array[4]).lower().split() for x in ['amount', 'credit', 'balance', 'debit', 'additions', 'subtractions']):
        columns = [x.lower() for x in data_array[4]]
        rows = data_array[5:]
        table_header_found = True
    else:
        columns = ['column_{}'.format(i) for i in range(len(data_array[0]))]
        rows = data_array
        table_header_found = False

    data=[]

    table_data = []

    col_name = bank_with_columns['keys'][0]['columns']
    col_index = 0

    row_lst =[]

    #import pdb; pdb.set_trace()
    ######################      table header found go with direct value extraction

    if table_header_found :
        for i, row in enumerate(rows):
            #print(row)
            row_dict = {}
            data.append({col_key: col for col, col_key in zip(row, columns)})

    elif not table_header_found and (area_type == "acc_summary" or area_type == "table_area"):
        for i, row in enumerate(rows):
            #print(row)
            row_dict = {}
            data.append({col_key: col for col, col_key in zip(row, columns)})

    #####################       table header not found and  generate generic table header based on regex
    elif not table_header_found and area_type != "table_area" :  #and area_type != "table_area"
        for i, row in enumerate(rows):
            #print(row)
            row_dict = {}
            for i,col_val in enumerate(row):
                val = ''.join(col_val.split(' '))
                #print(i, val)

                date_match = re.search(date_regex, val)
                amt_match = re.match(amount_regex, val)
                des_match = re.search(description_regex, val)
                num_match = re.match(number_regex, val)
                if date_match :
                    row_dict['date'] = val
                elif amt_match :
                    row_dict['amount_'+str(i)] = val
                elif des_match :
                    if 'description' in row_dict:
                        row_dict['description'] = row_dict['description'] + ' ' + col_val
                    else:
                        row_dict['description'] = col_val
                elif number_regex :
                    row_dict['check_num'] = val

            #print('row dictonary :::::::::: ')
            #print(row_dict)
            table_data.append(row_dict)

        #print(table_data)
        data= table_data   #######  generic table headers are generated


        #maxList = max(lst, key = lambda i: len(i))
        #maxLength = len(maxList)


        ########################  assign specific headers to generic headers according to bank statement type

        #if bank_statement_type == 1 : #BANK_OF_AMERICA

        #    print(' ')


    for i, row in enumerate(data):
        #import pdb; pdb.set_trace()
        if not row[columns[0]] :
            #import pdb; pdb.set_trace()
            data[i-1][columns[1]] += " " +row[columns[1]]


    data = [row for row in data if row[columns[0]]]

    #print('column separations == ', column_seperations)
    #print(columns)

    #import pdb; pdb.set_trace()

    return data, columns


def extract_tabular_data_using_sliding_window(vision_response, image, x1, x2, y1, y2,
                                              filename, bank_statement_type, area_type, doc_class, columns=[]):
    y1_modified = int(y1+((y2-y1)*0.27))
    y2_modified = int(y2-((y2-y1)*0.2))

    x1 = 0 if x1 < 0 else x1
    column_seperations = False

    if doc_class == 'USA_BANK_STATEMENTS':
        data, columns = extract_table_data_from_us_bs(vision_response, image, x1, x2, y1, y2, filename, bank_statement_type,area_type,doc_class)
        return data, columns

    else:
        column_seperations = get_columns_coordinates(image[y1:y2, x1:x2])

        if not column_seperations:
            updated_response = update_vision_response(vision_response, x1, x2, y1_modified, y2_modified)
            step1_remove_rows = remove_noise(image, updated_response, x1, x2, y1_modified, y2_modified)
            column_seperations = detect_column_seperations_using_sliding_window(cv2.cvtColor(step1_remove_rows, cv2.COLOR_BGR2GRAY),
                                                                                image, updated_response, x1, y1, y2)

        updated_response = update_vision_response(vision_response, x1, x2, y1, y2)
        lowers_ = get_straight_line_coordinates(updated_response, image.shape)
        sentence_dict, all_words = dict_sentences(updated_response, lowers_, image.shape[0], min_x=x1, min_y=y1, max_x=x2, max_y=y2)

        final_sentences = generate_final_sentence(sentence_dict)
        column_seperations.append(x2)

        column_seperations = sorted(column_seperations)

        data_array = [[''] * len(column_seperations[1:]) for x in lowers_]

        lowers_.insert(0, 0)
        for i, hs in enumerate(lowers_):
            for j, vs in enumerate(column_seperations):
                try:
                    z = get_between(all_words, min_x=vs, max_x=column_seperations[j + 1], min_y=hs,
                                    max_y=lowers_[i + 1])
                    data_array[i][j] = ' '.join(z)
                except IndexError:
                    pass

        if any(x in " ".join(data_array[0]).lower().split() for x in ['description', 'amount', 'credit', 'balance', 'particulars']):
            columns = [x.lower() for x in data_array[0]]
            rows = data_array[1:]
        elif any(x in " ".join(data_array[1]).lower().split() for x in ['description', 'amount', 'credit', 'balance', 'particulars']):
            columns = [x.lower() for x in data_array[1]]
            rows = data_array[2:]
        elif any(x in " ".join(data_array[2]).lower().split() for x in ['description', 'amount', 'credit', 'balance', 'particulars']):
            columns = [x.lower() for x in data_array[2]]
            rows = data_array[3:]
        elif any(x in " ".join(data_array[3]).lower().split() for x in ['description', 'amount', 'credit', 'balance', 'particulars']):
            columns = [x.lower() for x in data_array[3]]
            rows = data_array[4:]
        elif any(x in " ".join(data_array[4]).lower().split() for x in ['description', 'amount', 'credit', 'balance', 'particulars']):
            columns = [x.lower() for x in data_array[4]]
            rows = data_array[5:]
        else:
            if columns and len(data_array[0]) == len(columns):
                rows = data_array
            else:
                columns = ['column_{}'.format(i) for i in range(len(data_array[0]))]
                rows = data_array

        data = []
        for row in rows:
            data.append({col_key: col for col, col_key in zip(row, columns)})

        current = 0
        for column_key in ['particulars', 'description', 'narration']:
            if column_key in columns:
                for i, row in enumerate(data):
                    if row[columns[0]]:
                        current = i
                    else:
                        data[current][column_key] += " " + row[column_key]

        data = [row for row in data if row[columns[0]]]
        return data , columns


dt_three_digit_long_two_decimal_fmt_two_decimal_fmt = r"(\d{0,2}\/\d{0,2}\/\d{0,2})"


# ocr add decimal if not detected
def add_decimal_ocr_defect_for_non_detected(s):
    temp = s[::-1].strip()

    amt_1 = ''
    amt_2 = ''

    try:
        temp_holder = temp.split(' ')
        amt_1 = temp_holder[0]
        amt_2 = temp_holder[1]
    except:
        pass

    dec_point_find = 2

    if amt_1.find('.') == -1 and re.search(r"(-?\$?\(?\d+,?\.?\)?)", amt_1) is not None:
        if re.search(r"(-?\$?\(?\d+,?\.?\)?)", amt_1) is not None:
            amt_1 = re.search(r"(-?\$?\(?\d+,?\.?\)?)", amt_1).group()
        amt_1_temp = amt_1[:dec_point_find] + '.' + amt_1[dec_point_find:]
        amt_1_temp = amt_1_temp
    else:
        amt_1_temp = amt_1

    if amt_2.find('.') == -1 and re.search(r'(-?\$?\(?\d+,?\.?\)?)', amt_1) is not None and re.search(
            dt_three_digit_long_two_decimal_fmt_two_decimal_fmt, s) is None:
        if re.search(r"(-?\$?\(?\d+,?\.?\)?)", amt_2) is not None:
            amt_2 = re.search(r"(-?\$?\(?\d+,?\.?\)?)", amt_2).group()
        amt_2_temp = amt_2[:dec_point_find] + '.' + amt_2[dec_point_find:]
        amt_2_temp = amt_2_temp
    else:
        amt_2_temp = amt_2

    # else:
    #     return s.strip()

    # amt replace in temp
    if amt_1 != '':
        temp = temp.replace(amt_1, amt_1_temp)

    if amt_2 != '':
        temp = temp.replace(amt_2, amt_2_temp)

    return temp[::-1].strip()


# ocr comma defect remover
def replace_comma_decimal_ocr_defect(s):
    temp = s[::-1].strip()
    amt_1 = ''
    amt_2 = ''
    amt_1_temp = ''
    amt_2_temp = ''

    try:
        temp_holder = temp.split(' ')
        amt_1 = temp_holder[0]
        amt_2 = temp_holder[1]
    except:
        pass

    # if comma found on edge
    if amt_1.find(',') == 2:
        amt_1_temp = amt_1.replace(',', '.', 1)
    elif amt_2.find(',') == 2:
        amt_2_temp = amt_2.replace(',', '.', 1)

    # comma values correction - amt 1
    if re.search(r'(-?\$?\(?\d+,?\.?\)?)', amt_1) is not None and amt_1.count(',') > 1 and amt_1 != '':
        amt_1_temp = amt_1.replace(',', '.', 1)
    elif re.search(r'(-?\$?\(?\d+,?\.?\)?)', amt_1) is not None and amt_1.count('.') > 1 and amt_1 != '':
        amt_1_temp = amt_1[::-1].replace('.', ',', 1)[::-1]

    # comma values correction - amt 2
    if re.search(r'(-?\$?\(?\d+,?\.?\)?)', amt_2) is not None and amt_2.count(',') > 1 and amt_2 != '':
        amt_2_temp = amt_2.replace(',', '.', 1)
    elif re.search(r'(-?\$?\(?\d+,?\.?\)?)', amt_2) is not None and amt_2.count('.') > 1 and amt_2 != '':
        amt_2_temp = amt_2[::-1].replace('.', ',', 1)[::-1]

    # amt replace in temp
    if amt_1 != '':
        if amt_1_temp == '':
            amt_1_temp = amt_1
        temp = temp.replace(amt_1, amt_1_temp)

    if amt_2 != '':
        if amt_2_temp == '':
            amt_2_temp = amt_2
        temp = temp.replace(amt_2, amt_2_temp)

    return temp[::-1].strip()


def get_aggregate(vision_response, image_shape, confidence, final_sentences=False):
    document = vision_response['text_annotations']
    d_text = document[0]['description']
    form_b_agg  = ""

    if "TRACES" in d_text or "Certificate Number" in d_text or 'FORM No.12BA' in d_text or 'FORM NO.12BA' in d_text or 'ACES' in d_text:
        for doc in document:
            if doc['description'] in ['Aggregate']:
                length = doc['bounding_poly']['vertices'][1]['x'] - doc['bounding_poly']['vertices'][0]['x']
                width = doc['bounding_poly']['vertices'][3]['y'] - doc['bounding_poly']['vertices'][0]['y']

                x1 = doc['bounding_poly']['vertices'][1]['x'] + 12 * length
                y1 = doc['bounding_poly']['vertices'][1]['y']
                x2 = doc['bounding_poly']['vertices'][1]['x'] + 12 * length
                y2 = doc['bounding_poly']['vertices'][1]['y'] + width * 3

                form_b_agg, _ = text_within(vision_response['full_text_annotation'], x1, y1, x2, y2)

    if not form_b_agg:
        for doc in document:
            if doc['description'] in ['Aggregate']:
                length = doc['bounding_poly']['vertices'][1]['x'] - doc['bounding_poly']['vertices'][0]['x']
                width = doc['bounding_poly']['vertices'][3]['y'] - doc['bounding_poly']['vertices'][0]['y']

                x1 = doc['bounding_poly']['vertices'][1]['x'] + 10 * length
                y1 = doc['bounding_poly']['vertices'][1]['y']
                x2 = doc['bounding_poly']['vertices'][1]['x'] + 12 * length
                y2 = doc['bounding_poly']['vertices'][1]['y'] + width * 3

                form_b_agg, _ = text_within(vision_response['full_text_annotation'], x1, y1, x2, y2)

    if form_b_agg == "0 0" or form_b_agg == "0" or form_b_agg == ",000.00" or form_b_agg == "":
        if not final_sentences:
            lowers = get_straight_line_coordinates(document[1:], image_shape)
            final_sentences = generate_sentences(document[1:], lowers, int(image_shape[0]))

        for sentence in final_sentences:
            if any(x in sentence.replace(" ", "") for x in ["vi-a", "via"]):
                form_b_agg = sentence[sentence.find("vi") + len("vi") + 2:].strip().replace(" ", "").replace(",",
                                                                                                             "")
                if not form_b_agg.replace('.', '', 1).isdigit():
                    form_b_agg = ''

    return {'value': form_b_agg.replace("rs.", "").strip(), 'confidence': confidence}


def get_net_salary(vision_response, image_shape, confidence, final_sentences=False):
    if not final_sentences:
        lowers = get_straight_line_coordinates(vision_response['text_annotations'][1:], image_shape)
        final_sentences = generate_sentences(vision_response['text_annotations'][1:], lowers, int(image_shape[0]))

    form_b_agg_total_income = ""
    for i, sentence in enumerate(final_sentences):
        sentence = sentence.lower()
        if any(x in sentence.replace(" ", "").lower() for x in ['totalincomc(8-10)', 'totalincome(8-10)', 'totaltaxableincome(9-11']):
            temp_form_b_agg_total_income = sentence[sentence.find("income") + len("income") + 6:].strip().replace(" ", "").replace(",", "")
            match = re.search(r"\d{1,12}\.\d{2}|\d{1,12}", temp_form_b_agg_total_income)
            if match:
                form_b_agg_total_income = match.group()
            else:
                form_b_agg_total_income = ''

            if 'nearest' in form_b_agg_total_income:
                temp_form_b_agg_total_income = sentence[sentence.find("rupee") + len("rupee") + 1:].strip().replace(" ", "")
                match = re.search(r"\d{1,12}\.\d{2}|\d{1,12}", temp_form_b_agg_total_income)
                if match:
                    form_b_agg_total_income = match.group()
                else:
                    form_b_agg_total_income = ''

    if form_b_agg_total_income:
        return {'value': form_b_agg_total_income.replace("rs.", "").replace(',', ''), 'confidence': confidence}, final_sentences
    return {'value': '', 'confidence': ''}, final_sentences


def get_gross_salary_form16(vision_response, image_shape, confidence):
    lowers = get_straight_line_coordinates(vision_response['text_annotations'][1:], image_shape)
    final_sentences = generate_sentences(vision_response['text_annotations'][1:], lowers, int(image_shape[0]))

    gross_salary_found = False
    total_income, total_income_found = "", False

    for i, sentence in enumerate(final_sentences):
        sentence = sentence.lower()
        if not gross_salary_found and any(x in sentence.replace(" ", "") for x in ["grosssalary"]):
            gross_salary_found = True
            continue

        if not total_income_found and gross_salary_found and any(
            x in sentence.replace(" ", "") for x in ["(d)total", "d.total", 'total']):
            match = re.search(r"\d+.\d{2}", sentence)
            if match:
                total_income = match.group()

            # total_income = sentence[sentence.find("total") + len("total"):].strip().replace(" ", "").replace(",",
            # if not total_income.replace('.', '', 1).isdigit():
            #     total_income = ''
            break

    if total_income:
        return {'value': total_income.replace("rs.", ""), 'confidence': confidence}, final_sentences
    return {'value': '', 'confidence': ''}, final_sentences


def get_total_tax_payable(vision_response, image_shape, final_sentences=False):
    if not final_sentences:
        lowers = get_straight_line_coordinates(vision_response['text_annotations'][1:], image_shape)
        final_sentences = generate_sentences(vision_response['text_annotations'][1:], lowers, int(image_shape[0]))

    total_tax_payable, total_tax_payable_found = "", False

    for i, sentence in enumerate(final_sentences):
        sentence = sentence.lower()
        if not total_tax_payable_found and any(x in sentence.replace(" ", "") for x in
                                               ["taxpayable(12+13)", "taxpayable(14-15)", "taxpayable(15-16)",
                                                "taxpayable(16-17)",
                                                'nettaxpayable(']):
            total_tax_payable_found = True

            total_tax_payable = sentence[sentence.find("payable") + len("payable") + 8:].strip().replace(" ", "").replace(
                ",", "").replace("|", "")
            if not total_tax_payable.replace('.', '', 1).isdigit():
                total_tax_payable = ''
                total_tax_payable_found = False
                continue
            if total_tax_payable == '':
                total_tax_payable = final_sentences[i + 1].replace(" ", "")
                if not total_tax_payable.replace('.', '', 1).isdigit():
                    total_tax_payable_found = False
                    continue

        try:
            if not total_tax_payable_found and any(
                x in sentence.replace(" ", "") for x in ["(attachdetails)", "(attachdetails>"]):
                total_tax_payable_found = True
                total_tax_payable = sentence[sentence.find("details") + len("details") + 6:].strip().replace(" ",
                                                                                                             "")
                if not total_tax_payable.replace('.', '', 1).isdigit():
                    total_tax_payable = '0.00'
                    total_tax_payable_found = False
                    continue
                if total_tax_payable == '':
                    total_tax_payable = final_sentences[i + 1].replace(" ", "")
                    total_tax_payable_found = False
                    continue
        except:
            pass

    if total_tax_payable:
        return {'value': total_tax_payable.replace("rs.", "").replace(',', ''), 'confidence': "0.9557932615280151"}, final_sentences
    return {'value': '', 'confidence': ''}, final_sentences

#### Signature cropping for cheque
def crop_image(image, coordinates):
    if not coordinates:
        return None
    # cropped_image = Image.open(image)
    x1 = coordinates[1][0]
    x2 = coordinates[1][2]
    y1 = coordinates[0][3]
    y2 = coordinates[1][1]

    width = x2 - x1
    height = y2 - y1
    buffer = 0.05

    return store_image_on_google_drive(image[y1-int(height*buffer):y2+int(height*buffer),
                                       x1-int(width*buffer):x2+int(width*buffer)])
