from src.common.content import Content
from src.common.logging import logger
from src.tags.datastore.abstract import Datastore

logger = logger.bind(module="tag deletion")

def delete_tags_by_model(tagstore: Datastore, q: Content, model: str) -> int:
    batches = tagstore.find_batches(q, model=model, limit=100)

    # guard against the tagstore ignoring the model filter
    batches = [b for b in batches if b.model == model]

    for batch in batches:
        logger.debug(f"Deleting batch {batch.id} (model={model}, qid={q.qid})")
        tagstore.delete_batch(batch.id, q)

    return len(batches)
