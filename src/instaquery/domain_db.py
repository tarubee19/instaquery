"""Synthetic Indian quick-commerce database (dark stores, orders, inventory ...).

Used as an out-of-domain test: the fine-tuned model never saw this schema, so
execution accuracy here measures transfer to a realistic business database.
Generated deterministically (seed 42); all names/brands are fictional.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

CITY_AREAS = {
    "Bengaluru": [("Koramangala", "560034"), ("Indiranagar", "560038"), ("HSR Layout", "560102"),
                  ("Whitefield", "560066"), ("Jayanagar", "560041"), ("Electronic City", "560100"),
                  ("Marathahalli", "560037")],
    "Mumbai": [("Andheri West", "400053"), ("Bandra West", "400050"), ("Powai", "400076"),
               ("Lower Parel", "400013"), ("Thane West", "400601"), ("Malad West", "400064")],
    "Delhi": [("Saket", "110017"), ("Dwarka", "110075"), ("Rohini", "110085"), ("Lajpat Nagar", "110024"),
              ("Karol Bagh", "110005")],
    "Hyderabad": [("Gachibowli", "500032"), ("Madhapur", "500081"), ("Banjara Hills", "500034"),
                  ("Kukatpally", "500072")],
    "Pune": [("Kothrud", "411038"), ("Baner", "411045"), ("Viman Nagar", "411014"), ("Hinjewadi", "411057")],
    "Chennai": [("Adyar", "600020"), ("Velachery", "600042"), ("T Nagar", "600017"), ("Anna Nagar", "600040")],
}
CITY_CODE = {"Bengaluru": "BLR", "Mumbai": "MUM", "Delhi": "DEL", "Hyderabad": "HYD", "Pune": "PUN", "Chennai": "CHE"}
EXTRA_STORES = {"Koramangala": 3, "Andheri West": 2, "Gachibowli": 2, "Saket": 2}  # same-pincode clusters

CATALOG = {  # category -> (perishable, [(item, base_price)], brands)
    "Fruits & Vegetables": (1, [("Organic Bananas (6 pcs)", 69), ("Strawberries 200g", 149), ("Onion 1kg", 45),
                                ("Tomato 500g", 32), ("Potato 1kg", 39), ("Coriander Bunch", 15),
                                ("Shimla Apple 1kg", 189), ("Pomegranate 500g", 129), ("Spinach 250g", 29),
                                ("Avocado (2 pcs)", 249), ("Lemon (4 pcs)", 25)], ["FreshFarm", "GreenLeaf"]),
    "Dairy & Eggs": (1, [("Toned Milk 500ml", 28), ("Full Cream Milk 1L", 68), ("Curd 400g", 45), ("Paneer 200g", 90),
                         ("Butter 100g", 58), ("Cheese Slices 200g", 140), ("Brown Eggs (6 pcs)", 72),
                         ("Greek Yogurt 100g", 50)], ["DailyDairy", "MooFresh"]),
    "Bakery": (1, [("Whole Wheat Bread", 50), ("Multigrain Bread", 60), ("Pav (6 pcs)", 35), ("Croissant (2 pcs)", 99),
                   ("Chocolate Muffin", 45)], ["BakeHouse"]),
    "Snacks": (0, [("Classic Salted Chips 52g", 20), ("Masala Peanuts 200g", 60), ("Nachos 150g", 99),
                   ("Aloo Bhujia 400g", 110), ("Protein Bar", 120), ("Dark Chocolate 100g", 175),
                   ("Butter Cookies 250g", 85)], ["Crunchies", "SnackSutra"]),
    "Beverages": (0, [("Cola 750ml", 40), ("Orange Juice 1L", 125), ("Cold Coffee 200ml", 60), ("Green Tea 25 bags", 165),
                      ("Coconut Water 1L", 99), ("Energy Drink 250ml", 125), ("Packaged Water 1L", 20)],
                  ["SipWell", "ChillBox"]),
    "Staples": (0, [("Basmati Rice 1kg", 145), ("Toor Dal 1kg", 165), ("Whole Wheat Atta 5kg", 265),
                    ("Sunflower Oil 1L", 155), ("Sugar 1kg", 48), ("Iodised Salt 1kg", 28), ("Poha 500g", 45),
                    ("Besan 500g", 70)], ["AnnaPurna", "GharKa"]),
    "Personal Care": (0, [("Anti-Dandruff Shampoo 180ml", 199), ("Herbal Toothpaste 150g", 99),
                          ("Bath Soap (4 pcs)", 160), ("Face Wash 100ml", 175), ("Deodorant 150ml", 225)],
                      ["GlowUp", "PureCare"]),
    "Household": (0, [("Dishwash Liquid 500ml", 105), ("Detergent Powder 1kg", 130), ("Floor Cleaner 1L", 189),
                      ("Garbage Bags (30 pcs)", 99), ("Toilet Cleaner 500ml", 95)], ["CleanPro"]),
    "Baby Care": (0, [("Baby Diapers M (20 pcs)", 399), ("Baby Wipes (72 pcs)", 179)], ["LittleOne"]),
    "Frozen": (0, [("Frozen Green Peas 500g", 95), ("Vanilla Ice Cream Tub 700ml", 225), ("Frozen Paratha (5 pcs)", 110),
                   ("Chicken Nuggets 400g", 245)], ["FrostBite"]),
}
FIRST = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan", "Ananya",
         "Diya", "Aadhya", "Saanvi", "Pari", "Anika", "Navya", "Myra", "Riya", "Kavya", "Meera", "Neha", "Priya",
         "Rahul", "Karan", "Siddharth", "Nikhil", "Tanvi", "Pooja", "Sneha", "Harsh", "Yash", "Aisha", "Zoya", "Imran",
         "Farhan", "Gurpreet", "Simran", "Lakshmi", "Venkat"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Patel", "Nair", "Gupta", "Mehta", "Rao", "Singh", "Kulkarni", "Joshi",
        "Das", "Banerjee", "Menon", "Pillai", "Khan", "Shaikh", "Chopra", "Malhotra", "Agarwal", "Desai", "Naidu",
        "Bhat", "Kapoor"]

SCHEMA_SQL = """
CREATE TABLE dark_stores (
  store_id INTEGER PRIMARY KEY,
  store_name TEXT NOT NULL,
  city TEXT NOT NULL,
  area TEXT NOT NULL,
  pincode TEXT NOT NULL,
  opened_date TEXT NOT NULL,
  area_sqft INTEGER,
  daily_order_capacity INTEGER
);
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  product_name TEXT NOT NULL,
  category TEXT NOT NULL,
  brand TEXT NOT NULL,
  mrp REAL NOT NULL,
  selling_price REAL NOT NULL,
  is_perishable INTEGER NOT NULL
);
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  customer_name TEXT NOT NULL,
  city TEXT NOT NULL,
  pincode TEXT NOT NULL,
  signup_date TEXT NOT NULL,
  is_plus_member INTEGER NOT NULL
);
CREATE TABLE delivery_partners (
  partner_id INTEGER PRIMARY KEY,
  partner_name TEXT NOT NULL,
  store_id INTEGER NOT NULL,
  vehicle_type TEXT NOT NULL,
  rating REAL,
  joined_date TEXT,
  FOREIGN KEY (store_id) REFERENCES dark_stores(store_id)
);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  store_id INTEGER NOT NULL,
  partner_id INTEGER,
  order_time TEXT NOT NULL,
  status TEXT NOT NULL,
  payment_method TEXT NOT NULL,
  delivery_minutes INTEGER,
  order_value REAL NOT NULL,
  discount REAL NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (store_id) REFERENCES dark_stores(store_id),
  FOREIGN KEY (partner_id) REFERENCES delivery_partners(partner_id)
);
CREATE TABLE order_items (
  order_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL,
  PRIMARY KEY (order_id, product_id),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE inventory (
  store_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  stock_qty INTEGER NOT NULL,
  reorder_level INTEGER NOT NULL,
  PRIMARY KEY (store_id, product_id),
  FOREIGN KEY (store_id) REFERENCES dark_stores(store_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""


def _d(rng: random.Random, start: str, end: str) -> datetime:
    a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return a + timedelta(seconds=rng.randint(0, int((b - a).total_seconds())))


def build_quickcommerce_db(path: str | Path, n_customers: int = 3000, n_orders: int = 25000, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)

    # ---- stores
    stores = []
    sid = 1
    for city, areas in CITY_AREAS.items():
        for area, pin in areas:
            for k in range(1, 1 + EXTRA_STORES.get(area, 1)):
                name = f"{CITY_CODE[city]}-{area.replace(' ', '')}-{k:02d}"
                opened = _d(rng, "2022-01-01", "2025-12-31").date().isoformat()
                stores.append((sid, name, city, area, pin, opened, rng.randrange(2500, 6001, 100),
                               rng.randrange(600, 2001, 50)))
                sid += 1
    conn.executemany("INSERT INTO dark_stores VALUES (?,?,?,?,?,?,?,?)", stores)

    # ---- products (product names are unique: brand + item)
    products, pid = [], 1
    for cat, (perish, items, brands) in CATALOG.items():
        for item, base in items:
            for brand in brands[: 1 + (rng.random() < 0.45)]:
                mrp = round(base * rng.uniform(0.95, 1.15))
                sp = round(mrp * rng.uniform(0.80, 0.98), 2)
                products.append((pid, f"{brand} {item}", cat, brand, float(mrp), sp, perish))
                pid += 1
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", products)
    never_ordered = {p[0] for p in rng.sample(products, 5)}
    orderable = [p for p in products if p[0] not in never_ordered]
    weights = [3.0 if ("Bananas" in p[1] or "Milk" in p[1] or "Bread" in p[1] or "Onion" in p[1]) else 1.0
               for p in orderable]

    # ---- customers
    custs = []
    city_list = list(CITY_AREAS)
    for cid in range(1, n_customers + 1):
        city = rng.choices(city_list, weights=[30, 25, 18, 12, 8, 7])[0]
        area, pin = rng.choice(CITY_AREAS[city])
        custs.append((cid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", city, pin,
                      _d(rng, "2024-01-01", "2026-06-15").date().isoformat(), int(rng.random() < 0.25)))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", custs)

    # ---- delivery partners
    partners, ptid = [], 1
    by_store_partners: dict[int, list[int]] = {}
    for s in stores:
        for _ in range(rng.randint(6, 10)):
            vt = rng.choices(["bike", "scooter", "ev_scooter", "bicycle"], weights=[45, 25, 22, 8])[0]
            partners.append((ptid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", s[0], vt,
                             round(rng.uniform(3.4, 5.0), 1), _d(rng, "2023-01-01", "2026-05-31").date().isoformat()))
            by_store_partners.setdefault(s[0], []).append(ptid)
            ptid += 1
    conn.executemany("INSERT INTO delivery_partners VALUES (?,?,?,?,?,?)", partners)

    # ---- orders + items
    stores_by_pin: dict[str, list[int]] = {}
    stores_by_city: dict[str, list[int]] = {}
    slow_store = {s[0]: rng.uniform(-2, 6) for s in stores}
    for s in stores:
        stores_by_pin.setdefault(s[4], []).append(s[0])
        stores_by_city.setdefault(s[2], []).append(s[0])
    hours = list(range(24))
    hour_w = [1, 0.5, 0.3, 0.2, 0.2, 0.5, 2, 5, 8, 7, 5, 4, 4, 4, 3, 3, 4, 5, 8, 10, 9, 7, 4, 2]
    banana = next(p for p in orderable if "Organic Bananas" in p[1])
    berries = next(p for p in orderable if "Strawberries" in p[1])
    orders, items = [], []
    plus = {c[0]: c[5] for c in custs}
    for oid in range(1001, 1001 + n_orders):
        c = rng.choice(custs)
        cands = stores_by_pin.get(c[3]) or stores_by_city[c[2]]
        st = rng.choice(cands)
        day = _d(rng, "2026-01-01", "2026-06-30").date()
        hr = rng.choices(hours, weights=hour_w)[0]
        ts = datetime(day.year, day.month, day.day, hr, rng.randint(0, 59), rng.randint(0, 59))
        status = rng.choices(["delivered", "cancelled", "returned"], weights=[92, 6, 2])[0]
        pay = rng.choices(["UPI", "Card", "COD", "Wallet"], weights=[55, 20, 15, 10])[0]
        n_items = rng.choices([1, 2, 3, 4, 5, 6], weights=[15, 25, 25, 18, 10, 7])[0]
        basket = {}
        if rng.random() < 0.08:  # bananas + strawberries are frequently bought together
            basket[banana[0]] = banana
            basket[berries[0]] = berries
        while len(basket) < n_items:
            p = rng.choices(orderable, weights=weights)[0]
            basket[p[0]] = p
        value = 0.0
        for p in basket.values():
            q = rng.choices([1, 2, 3], weights=[70, 22, 8])[0]
            items.append((oid, p[0], q, p[5]))
            value += q * p[5]
        value = round(value, 2)
        discount = round(value * rng.choice([0.0, 0.05, 0.10]), 2) if plus[c[0]] else 0.0
        if status == "cancelled":
            mins, partner = None, None
        else:
            peak = 3 if hr in (8, 9, 18, 19, 20) else 0
            mins = int(max(5, min(45, rng.gauss(12 + peak + slow_store[st], 4))))
            partner = rng.choice(by_store_partners[st])
        orders.append((oid, c[0], st, partner, ts.strftime("%Y-%m-%d %H:%M:%S"), status, pay, mins, value, discount))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)

    # ---- inventory
    inv = []
    for s in stores:
        for p in products:
            reorder = rng.randrange(10, 41, 5)
            stock = 0 if rng.random() < 0.04 else rng.randint(0, 220)
            inv.append((s[0], p[0], stock, reorder))
    conn.executemany("INSERT INTO inventory VALUES (?,?,?,?)", inv)
    conn.commit()
    conn.close()
    return path
