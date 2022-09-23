Attribute VB_Name = "Module1"
Sub prep_workbook()
Attribute prep_workbook.VB_ProcData.VB_Invoke_Func = " \n14"
'
' test Macro
'
Sheets.Add.Name = "Data"
Sheets.Add.Name = "Structure"
Sheets("Table 1").Select
Cells.Select
Selection.Copy
Sheets("Data").Select
Selection.PasteSpecial Paste:=xlPasteValues, Operation:=xlNone, SkipBlanks _
    :=False, Transpose:=False
Range("A1").Select

End Sub

Sub findall()


Sheets("Data").Select
searchtext = "Summary of Existing Home Transactions"
    
Dim c As Range
Dim firstAddress As String
Dim found(200, 2) As String
i = 1
With Sheets("Data").Range("A1:zz10000")
    Set c = .Find(searchtext, LookIn:=xlValues, SearchDirection:=xlNext)
    If Not c Is Nothing Then
        firstAddress = c.Address
        Do
            found(i, 1) = c.Value
            found(i, 2) = c.Address
            i = i + 1
            
            Set c = .FindNext(c)
        Loop While Not c Is Nothing And c.Address <> firstAddress
    End If
End With


Sheets("Structure").Select
ActiveSheet.Range("A1").Select

For j = 1 To 100
    ActiveCell(j, 1) = (found(j, 1))
    ActiveCell(j, 2) = found(j, 2)
Next j


End Sub

