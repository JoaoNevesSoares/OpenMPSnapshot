/*
 * Copyright (c) 2019 Aman Nougrahiya, V Krishna Nandivada, IIT Madras.
 * This file is a part of the project IMOP, licensed under the MIT license.
 * See LICENSE.md for the full text of the license.
 * 
 * The above notice shall be included in all copies or substantial
 * portions of this file.
 */
/* Generated By:JavaCC: Do not edit this line. CParserConstants.java */
package imop.parser;

/**
 * Token literal values and constants.
 * Generated by org.javacc.parser.OtherFilesGen#start()
 */
public interface CParserConstants {

	/** End of File. */
	int EOF = 0;
	/** RegularExpression Id. */
	int INTEGER_LITERAL = 8;
	/** RegularExpression Id. */
	int DECIMAL_LITERAL = 9;
	/** RegularExpression Id. */
	int HEX_LITERAL = 10;
	/** RegularExpression Id. */
	int OCTAL_LITERAL = 11;
	/** RegularExpression Id. */
	int FLOATING_POINT_LITERAL = 12;
	/** RegularExpression Id. */
	int EXPONENT = 13;
	/** RegularExpression Id. */
	int CHARACTER_LITERAL = 14;
	/** RegularExpression Id. */
	int STRING_LITERAL = 15;
	/** RegularExpression Id. */
	int RESTRICT = 23;
	/** RegularExpression Id. */
	int CONTINUE = 24;
	/** RegularExpression Id. */
	int VOLATILE = 25;
	/** RegularExpression Id. */
	int REGISTER = 26;
	/** RegularExpression Id. */
	int CCONST = 27;
	/** RegularExpression Id. */
	int INLINE = 28;
	/** RegularExpression Id. */
	int CINLINED = 29;
	/** RegularExpression Id. */
	int CINLINED2 = 30;
	/** RegularExpression Id. */
	int CSIGNED = 31;
	/** RegularExpression Id. */
	int CSIGNED2 = 32;
	/** RegularExpression Id. */
	int UNSIGNED = 33;
	/** RegularExpression Id. */
	int TYPEDEF = 34;
	/** RegularExpression Id. */
	int DFLT = 35;
	/** RegularExpression Id. */
	int DOUBLE = 36;
	/** RegularExpression Id. */
	int SWITCH = 37;
	/** RegularExpression Id. */
	int RETURN = 38;
	/** RegularExpression Id. */
	int EXTERN = 39;
	/** RegularExpression Id. */
	int STRUCT = 40;
	/** RegularExpression Id. */
	int STATIC = 41;
	/** RegularExpression Id. */
	int SIGNED = 42;
	/** RegularExpression Id. */
	int WHILE = 43;
	/** RegularExpression Id. */
	int BREAK = 44;
	/** RegularExpression Id. */
	int UNION = 45;
	/** RegularExpression Id. */
	int CONST = 46;
	/** RegularExpression Id. */
	int FLOAT = 47;
	/** RegularExpression Id. */
	int SHORT = 48;
	/** RegularExpression Id. */
	int ELSE = 49;
	/** RegularExpression Id. */
	int CASE = 50;
	/** RegularExpression Id. */
	int LONG = 51;
	/** RegularExpression Id. */
	int ENUM = 52;
	/** RegularExpression Id. */
	int AUTO = 53;
	/** RegularExpression Id. */
	int VOID = 54;
	/** RegularExpression Id. */
	int CHAR = 55;
	/** RegularExpression Id. */
	int GOTO = 56;
	/** RegularExpression Id. */
	int FOR = 57;
	/** RegularExpression Id. */
	int INT = 58;
	/** RegularExpression Id. */
	int IF = 59;
	/** RegularExpression Id. */
	int DO = 60;
	/** RegularExpression Id. */
	int SIZEOF = 61;
	/** RegularExpression Id. */
	int EXTENSION = 62;
	/** RegularExpression Id. */
	int CATOMIC = 63;
	/** RegularExpression Id. */
	int COMPLEX = 64;
	/** RegularExpression Id. */
	int ELLIPSIS = 65;
	/** RegularExpression Id. */
	int OP_SLASS = 66;
	/** RegularExpression Id. */
	int OP_SRASS = 67;
	/** RegularExpression Id. */
	int OP_EQ = 68;
	/** RegularExpression Id. */
	int OP_AND = 69;
	/** RegularExpression Id. */
	int OP_OR = 70;
	/** RegularExpression Id. */
	int OP_MULASS = 71;
	/** RegularExpression Id. */
	int OP_DIVASS = 72;
	/** RegularExpression Id. */
	int OP_MODASS = 73;
	/** RegularExpression Id. */
	int OP_ADDASS = 74;
	/** RegularExpression Id. */
	int OP_SUBASS = 75;
	/** RegularExpression Id. */
	int OP_ANDASS = 76;
	/** RegularExpression Id. */
	int OP_XORASS = 77;
	/** RegularExpression Id. */
	int OP_ORASS = 78;
	/** RegularExpression Id. */
	int OP_SL = 79;
	/** RegularExpression Id. */
	int OP_SR = 80;
	/** RegularExpression Id. */
	int OP_NEQ = 81;
	/** RegularExpression Id. */
	int OP_GE = 82;
	/** RegularExpression Id. */
	int OP_LE = 83;
	/** RegularExpression Id. */
	int OP_DEREF = 84;
	/** RegularExpression Id. */
	int OP_INCR = 85;
	/** RegularExpression Id. */
	int OP_DECR = 86;
	/** RegularExpression Id. */
	int OP_GT = 87;
	/** RegularExpression Id. */
	int OP_LT = 88;
	/** RegularExpression Id. */
	int OP_ADD = 89;
	/** RegularExpression Id. */
	int OP_SUB = 90;
	/** RegularExpression Id. */
	int OP_MUL = 91;
	/** RegularExpression Id. */
	int OP_DIV = 92;
	/** RegularExpression Id. */
	int OP_MOD = 93;
	/** RegularExpression Id. */
	int OP_ASS = 94;
	/** RegularExpression Id. */
	int OP_BITAND = 95;
	/** RegularExpression Id. */
	int OP_BITOR = 96;
	/** RegularExpression Id. */
	int OP_BITXOR = 97;
	/** RegularExpression Id. */
	int OP_NOT = 98;
	/** RegularExpression Id. */
	int OP_BITNOT = 99;
	/** RegularExpression Id. */
	int COLON = 100;
	/** RegularExpression Id. */
	int SEMICOLON = 101;
	/** RegularExpression Id. */
	int QUESTION = 102;
	/** RegularExpression Id. */
	int DOT = 103;
	/** RegularExpression Id. */
	int LEFTPAREN = 104;
	/** RegularExpression Id. */
	int RIGHTPAREN = 105;
	/** RegularExpression Id. */
	int LEFTBRACKET = 106;
	/** RegularExpression Id. */
	int RIGHTBRACKET = 107;
	/** RegularExpression Id. */
	int LEFTBRACE = 108;
	/** RegularExpression Id. */
	int RIGHTBRACE = 109;
	/** RegularExpression Id. */
	int COMMA = 110;
	/** RegularExpression Id. */
	int CROSSBAR = 111;
	/** RegularExpression Id. */
	int UNKNOWN_CPP = 112;
	/** RegularExpression Id. */
	int PRAGMA = 113;
	/** RegularExpression Id. */
	int OMP_NL = 134;
	/** RegularExpression Id. */
	int OMP_CR = 135;
	/** RegularExpression Id. */
	int OMP = 140;
	/** RegularExpression Id. */
	int PARALLEL = 142;
	/** RegularExpression Id. */
	int SECTIONS = 143;
	/** RegularExpression Id. */
	int SECTION = 144;
	/** RegularExpression Id. */
	int SINGLE = 145;
	/** RegularExpression Id. */
	int ORDERED = 146;
	/** RegularExpression Id. */
	int MASTER = 147;
	/** RegularExpression Id. */
	int CRITICAL = 148;
	/** RegularExpression Id. */
	int ATOMIC = 149;
	/** RegularExpression Id. */
	int BARRIER = 150;
	/** RegularExpression Id. */
	int FLUSH = 151;
	/** RegularExpression Id. */
	int NOWAIT = 152;
	/** RegularExpression Id. */
	int SCHEDULE = 153;
	/** RegularExpression Id. */
	int DYNAMIC = 154;
	/** RegularExpression Id. */
	int GUIDED = 155;
	/** RegularExpression Id. */
	int RUNTIME = 156;
	/** RegularExpression Id. */
	int NONE = 157;
	/** RegularExpression Id. */
	int REDUCTION = 158;
	/** RegularExpression Id. */
	int PRIVATE = 159;
	/** RegularExpression Id. */
	int FIRSTPRIVATE = 160;
	/** RegularExpression Id. */
	int LASTPRIVATE = 161;
	/** RegularExpression Id. */
	int COPYPRIVATE = 162;
	/** RegularExpression Id. */
	int SHARED = 163;
	/** RegularExpression Id. */
	int COPYIN = 164;
	/** RegularExpression Id. */
	int THREADPRIVATE = 165;
	/** RegularExpression Id. */
	int NUM_THREADS = 166;
	/** RegularExpression Id. */
	int COLLAPSE = 167;
	/** RegularExpression Id. */
	int READ = 168;
	/** RegularExpression Id. */
	int WRITE = 169;
	/** RegularExpression Id. */
	int UPDATE = 170;
	/** RegularExpression Id. */
	int CAPTURE = 171;
	/** RegularExpression Id. */
	int TASK = 172;
	/** RegularExpression Id. */
	int TASKWAIT = 173;
	/** RegularExpression Id. */
	int DECLARE = 174;
	/** RegularExpression Id. */
	int TASKYIELD = 175;
	/** RegularExpression Id. */
	int UNTIED = 176;
	/** RegularExpression Id. */
	int MERGEABLE = 177;
	/** RegularExpression Id. */
	int INITIALIZER = 178;
	/** RegularExpression Id. */
	int FINAL = 179;
	/** RegularExpression Id. */
	int IDENTIFIER = 180;
	/** RegularExpression Id. */
	int LETTER = 181;
	/** RegularExpression Id. */
	int DIGIT = 182;

	/** Lexical state. */
	int DEFAULT = 0;
	/** Lexical state. */
	int AfterCrossbar = 1;
	/** Lexical state. */
	int Pragma = 2;
	/** Lexical state. */
	int Omp = 3;
	/** Lexical state. */
	int AfterAttrib = 4;
	/** Lexical state. */
	int Cpp = 5;

	/** Literal token values. */
	String[] tokenImage = { "<EOF>", "\" \"", "\"\\t\"", "\"\\n\"", "\"\\r\"", "\"\\f\"", "<token of kind 6>",
			"<token of kind 7>", "<INTEGER_LITERAL>", "<DECIMAL_LITERAL>", "<HEX_LITERAL>", "<OCTAL_LITERAL>",
			"<FLOATING_POINT_LITERAL>", "<EXPONENT>", "<CHARACTER_LITERAL>", "<STRING_LITERAL>", "\"__attribute__\"",
			"\"__asm\"", "\"__asm__\"", "\"asm\"", "\"(\"", "\")\"", "<token of kind 22>", "<RESTRICT>", "\"continue\"",
			"\"volatile\"", "\"register\"", "\"__const\"", "\"inline\"", "\"__inline\"", "\"__inline__\"",
			"\"__signed\"", "\"__signed__\"", "\"unsigned\"", "\"typedef\"", "\"default\"", "\"double\"", "\"switch\"",
			"\"return\"", "\"extern\"", "\"struct\"", "\"static\"", "\"signed\"", "\"while\"", "\"break\"", "\"union\"",
			"\"const\"", "\"float\"", "\"short\"", "\"else\"", "\"case\"", "\"long\"", "\"enum\"", "\"auto\"",
			"\"void\"", "\"char\"", "\"goto\"", "\"for\"", "\"int\"", "\"if\"", "\"do\"", "\"sizeof\"",
			"\"__extension__\"", "\"_Atomic\"", "\"_Complex\"", "\"...\"", "\"<<=\"", "\">>=\"", "\"==\"", "\"&&\"",
			"\"||\"", "\"*=\"", "\"/=\"", "\"%=\"", "\"+=\"", "\"-=\"", "\"&=\"", "\"^=\"", "\"|=\"", "\"<<\"",
			"\">>\"", "\"!=\"", "\">=\"", "\"<=\"", "\"->\"", "\"++\"", "\"--\"", "\">\"", "\"<\"", "\"+\"", "\"-\"",
			"\"*\"", "\"/\"", "\"%\"", "\"=\"", "\"&\"", "\"|\"", "\"^\"", "\"!\"", "\"~\"", "\":\"", "\";\"", "\"?\"",
			"\".\"", "\"(\"", "\")\"", "\"[\"", "\"]\"", "\"{\"", "\"}\"", "\",\"", "\"#\"", "<UNKNOWN_CPP>",
			"\"pragma\"", "\"include\"", "\"import\"", "\"define\"", "\"ifndef\"", "\"ident\"", "\"undef\"",
			"\"ifdef\"", "\"endif\"", "\"line\"", "\"else\"", "\"if\"", "\"elif\"", "<token of kind 126>",
			"<token of kind 127>", "\" \"", "\"\\t\"", "<token of kind 130>", "<token of kind 131>", "\"\\n\"",
			"\"\\r\"", "\"\\n\"", "\"\\r\"", "\" \"", "\"\\t\"", "<token of kind 138>", "<token of kind 139>",
			"\"omp\"", "<token of kind 141>", "\"parallel\"", "\"sections\"", "\"section\"", "\"single\"",
			"\"ordered\"", "\"master\"", "\"critical\"", "\"atomic\"", "\"barrier\"", "\"flush\"", "\"nowait\"",
			"\"schedule\"", "\"dynamic\"", "\"guided\"", "\"runtime\"", "\"none\"", "\"reduction\"", "\"private\"",
			"\"firstprivate\"", "\"lastprivate\"", "\"copyprivate\"", "\"shared\"", "\"copyin\"", "\"threadprivate\"",
			"\"num_threads\"", "\"collapse\"", "\"read\"", "\"write\"", "\"update\"", "\"capture\"", "\"task\"",
			"\"taskwait\"", "\"declare\"", "\"taskyield\"", "\"untied\"", "\"mergeable\"", "\"initializer\"",
			"\"final\"", "<IDENTIFIER>", "<LETTER>", "<DIGIT>", };

}