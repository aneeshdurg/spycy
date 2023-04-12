tck_List
           :  '[' SP? ( tck_ExpectedValue SP? ( ',' SP? tck_ExpectedValue SP? )* )? ']' ;

tck_Map
          :  '{' SP? ( oC_PropertyKeyName SP? ':' SP? tck_ExpectedValue SP? ( ',' SP? oC_PropertyKeyName SP? ':' SP? tck_ExpectedValue SP? )* )? '}' ;

tck_ExpectedValue
          : ( tck_Map | tck_List | oC_Literal | oC_Pattern | oC_RelationshipDetail ) ;
