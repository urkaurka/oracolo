\d+ post_post


\d+ post_source

SELECT post.ID
    ,post.post_title AS Titolo
    ,post.post_content AS Contenuto
    ,GROUP_CONCAT(termsname.`name` SEPARATOR ',') AS Categorie
FROM UmCUCQAf0_posts post
LEFT JOIN (
    SELECT terms.name
        ,relat.object_id
    FROM UmCUCQAf0_term_relationships relat
    LEFT JOIN UmCUCQAf0_terms terms ON relat.term_taxonomy_id = terms.term_id
    ) termsname ON termsname.object_id = post.ID
WHERE post.post_type = 'post'
GROUP BY post.ID


select extract_text
from post_source
limit 10
