### What fine-tuning changed (spider_dev)

* Questions the base model got wrong and the fine-tuned model got right: **124**
* Questions the base model got right and the fine-tuned model got wrong: **66**

| Error type | Base, zero-shot | Fine-tuned (QLoRA) |
|---|---|---|
| unknown column | 140 | 73 |
| wrong result (row count) | 104 | 103 |
| wrong result (values) | 73 | 87 |
| other execution error | 5 | 4 |
| unknown table | 4 | 1 |

Examples fixed by fine-tuning:

* **Q:** Show the name and the release year of the song by the youngest singer.  
  base: `SELECT T2.Song_Name, T2.Song_release_year FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age ASC LIMIT 1` (unknown column)  
  fine-tuned: `SELECT song_name , song_release_year FROM singer ORDER BY age LIMIT 1`
* **Q:** What are the names and release years for all the songs of the youngest singer?  
  base: `SELECT T2.Song_Name, T2.Song_release_year FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID ORDER BY T1.Age ASC LIMIT 1` (unknown column)  
  fine-tuned: `SELECT song_name , song_release_year FROM singer ORDER BY age LIMIT 1`
* **Q:** List all song names by singers above the average age.  
  base: `SELECT T2.song_name FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id WHERE T1.age > (SELECT AVG(age) FROM singer)` (unknown column)  
  fine-tuned: `SELECT song_name FROM singer WHERE age > (SELECT avg(age) FROM singer)`
* **Q:** Show countries where a singer above age 40 and a singer below 30 are from.  
  base: `SELECT DISTINCT T2.Country FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID WHERE T1.Age > 40 INTERSECT SELECT DISTINCT T2.Country FROM singer AS T1 INNER JOIN singer_in_concert AS T2 O` (unknown column)  
  fine-tuned: `SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30`
* **Q:** what is the name and nation of the singer who have a song having 'Hey' in its name?  
  base: `SELECT T1.Name, T1.Country FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID INNER JOIN concert AS T3 ON T2.concert_ID = T3.concert_ID WHERE T3.Theme LIKE '%Hey%'` (wrong result (row count))  
  fine-tuned: `SELECT name , country FROM singer WHERE song_name LIKE "%Hey%"`