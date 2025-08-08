package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

type FunctionInfo struct {
	Name       string   `json:"name"`
	StartLine  int      `json:"start_line"`
	EndLine    int      `json:"end_line"`
	Parameters []string `json:"parameters"`
	Returns    string   `json:"returns"`
	Calls      []string `json:"calls"`
	IsMethod   bool     `json:"is_method"`
	Receiver   string   `json:"receiver"`
	DocString  string   `json:"docstring"`
	RawCode    string   `json:"raw_code"`
}

type FileInfo struct {
	Functions []FunctionInfo `json:"functions"`
	Imports   []string       `json:"imports"`
}

// extractFunctionCalls returns function calls inside the node
func extractFunctionCalls(node ast.Node) []string {
	calls := make(map[string]bool)

	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			switch fun := x.Fun.(type) {
			case *ast.Ident:
				calls[fun.Name] = true
			case *ast.SelectorExpr:
				calls[fun.Sel.Name] = true
			}
		}
		return true
	})

	result := make([]string, 0, len(calls))
	for call := range calls {
		result = append(result, call)
	}
	return result
}

// extractParameters returns the parameter types
// extractParameters returns the parameter types
func extractParameters(params *ast.FieldList) []string {
	if params == nil {
		return []string{}
	}

	var result []string
	for _, param := range params.List {
		paramType := extractTypeString(param.Type)

		// Handle multiple parameter names with the same type (e.g., "a, b int")
		if len(param.Names) == 0 {
			// Anonymous parameter
			result = append(result, paramType)
		} else {
			// Named parameters - each name gets the same type
			for range param.Names {
				result = append(result, paramType)
			}
		}
	}
	return result
}

// extractTypeString converts an ast.Expr representing a type to its string repr
func extractTypeString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name

	case *ast.StarExpr:
		return "*" + extractTypeString(t.X)

	case *ast.ArrayType:
		if t.Len == nil {
			// Slice
			return "[]" + extractTypeString(t.Elt)
		}
		// Array -- for simplicity, we'll show it as []type
		return "[]" + extractTypeString(t.Elt)

	case *ast.MapType:
		return "map[" + extractTypeString(t.Key) + "]" + extractTypeString(t.Value)

	case *ast.ChanType:
		switch t.Dir {
		case ast.SEND:
			return "chan<- " + extractTypeString(t.Value)
		case ast.RECV:
			return "<-chan " + extractTypeString(t.Value)
		default:
			return "chan " + extractTypeString(t.Value)
		}

	case *ast.FuncType:
		return "func" // Simplified - could be expanded to show full signature

	case *ast.InterfaceType:
		if len(t.Methods.List) == 0 {
			return "interface{}"
		}
		return "interface{...}" // Simplified

	case *ast.StructType:
		return "struct{...}" // Simplified

	case *ast.SelectorExpr:
		if x, ok := t.X.(*ast.Ident); ok {
			return x.Name + "." + t.Sel.Name
		}
		return "unknown.selector"

	case *ast.Ellipsis:
		return "..." + extractTypeString(t.Elt)

	default:
		return "unknown"
	}
}

// extractReturnTypes returns return types
func extractReturnTypes(results *ast.FieldList) string {
	if results == nil {
		return ""
	}

	var types []string
	for _, result := range results.List {
		switch t := result.Type.(type) {
		case *ast.Ident:
			types = append(types, t.Name)
		case *ast.SelectorExpr:
			if x, ok := t.X.(*ast.Ident); ok {
				types = append(types, x.Name+"."+t.Sel.Name)
			}
		default:
			types = append(types, "unknown")
		}
	}
	return strings.Join(types, ", ")
}

// extractDocstring returns the docstring cleaned up a bit
func extractDocstring(cg *ast.CommentGroup) string {
	if cg == nil {
		return ""
	}

	var lines []string
	for _, comment := range cg.List {
		line := strings.TrimPrefix(comment.Text, "//")
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	return strings.Join(lines, " ")
}

// extractImports returns the imports
func extractImports(file *ast.File) []string {
	var imports []string

	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, "\"")
		if imp.Name != nil {
			imports = append(imports, imp.Name.Name+" "+path)
		} else {
			imports = append(imports, path)
		}
	}
	return imports
}

func main() {
    if len(os.Args) != 2 {
        fmt.Fprintf(os.Stderr, "Usage: %s <go-file>\n", os.Args[0])
        os.Exit(1)
    }

    filename := os.Args[1]

    content, err := os.ReadFile(filename)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
        os.Exit(1)
    }
    sourceLines := strings.Split(string(content), "\n")

    fSet := token.NewFileSet()
    node, err := parser.ParseFile(fSet, filename, nil, parser.ParseComments)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error parsing file: %v\n", err)
        os.Exit(1)
    }

    fileInfo := FileInfo{
        Functions: []FunctionInfo{},
        Imports:   extractImports(node),
    }

    ast.Inspect(node, func(n ast.Node) bool {
        switch x := n.(type) {
        case *ast.FuncDecl:
            if x.Name.IsExported() || strings.HasPrefix(x.Name.Name, "_") || x.Name.Name != "_" {
                startPos := fSet.Position(x.Pos())
                endPos := fSet.Position(x.End())

                receiver := ""
                isMethod := false
                if x.Recv != nil && len(x.Recv.List) > 0 {
                    isMethod = true
                    switch t := x.Recv.List[0].Type.(type) {
                    case *ast.Ident:
                        receiver = t.Name
                    case *ast.StarExpr:
                        if ident, ok := t.X.(*ast.Ident); ok {
                            receiver = "*" + ident.Name
                        }
                    }
                }

                rawCode := ""
                if startPos.Line > 0 && endPos.Line > 0 && startPos.Line <= len(sourceLines) && endPos.Line <= len(sourceLines) {
                    funcLines := sourceLines[startPos.Line-1:endPos.Line]
                    rawCode = strings.Join(funcLines, "\n")
                }

                funcInfo := FunctionInfo{
                    Name:       x.Name.Name,
                    StartLine:  startPos.Line,
                    EndLine:    endPos.Line,
                    Parameters: extractParameters(x.Type.Params),
                    Returns:    extractReturnTypes(x.Type.Results),
                    Calls:      extractFunctionCalls(x),
                    IsMethod:   isMethod,
                    Receiver:   receiver,
                    DocString:  extractDocstring(x.Doc),
                    RawCode:    rawCode,
                }

                fileInfo.Functions = append(fileInfo.Functions, funcInfo)
            }
        }
        return true
    })

    output, err := json.Marshal(fileInfo)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error marshaling JSON: %v\n", err)
        os.Exit(1)
    }

    fmt.Println(string(output))
}
