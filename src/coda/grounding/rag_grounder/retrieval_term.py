from typing import List, Optional
from gilda import Term


class RetrievalTerm(Term):
    """
    A term in the retrieval space with identifier, name, synonyms, and definition.
    
    Extends gilda.Term to add fields needed for RAG-based grounding.
    """
    
    def __init__(
        self,
        norm_text: str,
        text: str,
        db: str,
        id: str,
        entry_name: str,
        status: str,
        source: str,
        organism: Optional[str] = None,
        source_db: Optional[str] = None,
        source_id: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        definition: Optional[str] = None,
    ):
        """
        Initialize retrieval term.
        
        Parameters
        ----------
        norm_text : str
            Normalized text for lookups (from gilda.Term).
        text : str
            The text entry itself (from gilda.Term).
        db : str
            Database/namespace (from gilda.Term).
        id : str
            Identifier within the database (from gilda.Term).
        entry_name : str
            Standardized name (from gilda.Term).
        status : str
            Relationship of text entry to grounded term (from gilda.Term).
        source : str
            Source from which term was obtained (from gilda.Term).
        organism : str, optional
            Taxonomy code for proteins (from gilda.Term).
        source_db : str, optional
            Original db before mapping (from gilda.Term).
        source_id : str, optional
            Original id before mapping (from gilda.Term).
        synonyms : List[str], optional
            List of alternative names/synonyms. Defaults to empty list.
        definition : str, optional
            Description or definition of the term. Defaults to empty string.
        """
        super().__init__(
            norm_text=norm_text,
            text=text,
            db=db,
            id=id,
            entry_name=entry_name,
            status=status,
            source=source,
            organism=organism,
            source_db=source_db,
            source_id=source_id
        )
        
        # Store retrieval-specific fields
        self.synonyms = synonyms or []
        self.definition = definition
