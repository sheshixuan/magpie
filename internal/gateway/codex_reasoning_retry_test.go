package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// A real backend names only the first unreadable item. Six sealed items
// from another provider used to exhaust the native path's three retries.
func TestCodexOwnModelRecoversSeveralSealedItems(t *testing.T) {
	for _, named := range []bool{true, false} {
		t.Run(fmt.Sprintf("named=%t", named), func(t *testing.T) {
			setup(t, provider.Chat, &fake{t: t})
			var got [][]json.RawMessage
			var mu sync.Mutex
			chatgpt(t, func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				var q struct {
					Input []json.RawMessage `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
					t.Error(err)
					w.WriteHeader(500)
					return
				}
				got = append(got, q.Input)
				for _, raw := range q.Input {
					var it struct{ Type, ID string }
					json.Unmarshal(raw, &it)
					if it.Type == "reasoning" {
						message := "Encrypted content could not be decrypted or parsed."
						if named {
							message = "The encrypted content for item " + it.ID + " could not be verified. Reason: " + message
						}
						w.WriteHeader(400)
						json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": "invalid_encrypted_content", "message": message}})
						return
					}
				}
				io.WriteString(w, sse(`data: {"type":"response.completed","response":{"id":"r1"}}`))
			})
			input := []json.RawMessage{json.RawMessage(`{"type":"message","role":"user","content":[{"type":"input_text","text":"Keep the theme green"}]}`)}
			for i := range 6 {
				input = append(input, json.RawMessage(fmt.Sprintf(`{"type":"reasoning","id":"rs_foreign_%d","content":null,"summary":[{"type":"summary_text","text":"Decision %d"},{"type":"summary_text","text":"Keep context"}],"encrypted_content":"synthetic-seal-%d"}`, i, i, i)))
			}
			input = append(input,
				json.RawMessage(`{"type":"reasoning","id":"rs_empty","summary":[],"encrypted_content":"synthetic-empty"}`),
				json.RawMessage(`{"type":"function_call","id":"fc_test","call_id":"call_test","name":"lookup","arguments":"{}"}`),
				json.RawMessage(`{"type":"function_call_output","call_id":"call_test","output":"green"}`),
				json.RawMessage(`{"type":"compaction","id":"cmp_test","encrypted_content":"opaque-checkpoint"}`),
				json.RawMessage(`{"type":"message","role":"user","content":[{"type":"input_text","text":"Continue"}]}`))
			body, _ := json.Marshal(map[string]any{"model": "gpt-5.5", "stream": true, "input": input})
			before := bytes.Clone(body)
			gateway := httptest.NewServer(lanGuard(New().Handler()))
			defer gateway.Close()
			req, _ := http.NewRequest("POST", gateway.URL+CodexPath+"/responses", bytes.NewReader(body))
			req.Header.Set("Authorization", "Bearer synthetic-chatgpt-token")
			req.Header.Set("chatgpt-account-id", "synthetic-account")
			res, err := gateway.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			answer, _ := io.ReadAll(res.Body)
			mu.Lock()
			defer mu.Unlock()
			if res.StatusCode != 200 || !bytes.Contains(answer, []byte("response.completed")) {
				t.Fatalf("status=%d requests=%d body=%s", res.StatusCode, len(got), answer)
			}
			if len(got) != 2 || len(got[0]) != len(input) || len(got[1]) != len(input)-1 {
				t.Fatalf("requests: %s", got)
			}
			for i, raw := range got[0] {
				if !bytes.Equal(raw, input[i]) {
					t.Fatalf("initial item %d changed: %s", i, raw)
				}
			}
			for i := range 6 {
				var message struct {
					Type, Role string
					Content    []struct{ Type, Text string }
				}
				raw := got[1][i+1]
				if err := json.Unmarshal(raw, &message); err != nil {
					t.Fatal(err)
				}
				if message.Type != "message" || message.Role != "assistant" || len(message.Content) != 1 ||
					message.Content[0].Type != "output_text" || message.Content[0].Text != fmt.Sprintf("Decision %d\n\nKeep context", i) ||
					bytes.Contains(raw, []byte("encrypted_content")) || bytes.Contains(raw, []byte("rs_foreign")) {
					t.Errorf("summary %d: %s", i, raw)
				}
			}
			for _, pair := range [][2]int{{0, 0}, {8, 7}, {9, 8}, {10, 9}, {11, 10}} {
				if !bytes.Equal(input[pair[0]], got[1][pair[1]]) {
					t.Errorf("nonreasoning item changed: %s", got[1][pair[1]])
				}
			}
			if !bytes.Equal(body, before) {
				t.Error("client request mutated")
			}
		})
	}
}

func TestCodexUnreadableDoesNotBroadenOtherErrors(t *testing.T) {
	body := []byte(`{"input":[{"type":"reasoning","id":"rs_one","encrypted_content":"sealed","summary":[{"type":"summary_text","text":"context"}]},{"type":"compaction","id":"cmp_one","encrypted_content":"checkpoint"}]}`)
	for _, tc := range []struct {
		name, message                  string
		changed, reasoning, compaction bool
	}{
		{"unrelated", `{"error":{"message":"invalid argument"}}`, false, true, true},
		{"not found", `{"error":{"message":"Item with id 'rs_one' not found"}}`, true, false, true},
		{"checkpoint", `{"error":{"code":"invalid_encrypted_content","message":"The encrypted content for item cmp_one could not be verified"}}`, true, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, changed := withoutUnreadable(body, []byte(tc.message))
			if changed != tc.changed {
				t.Fatalf("changed=%t body=%s", changed, got)
			}
			if !changed {
				return
			}
			if strings.Contains(string(got), `"reasoning"`) != tc.reasoning || strings.Contains(string(got), `"compaction"`) != tc.compaction {
				t.Errorf("unexpected recovery: %s", got)
			}
		})
	}
}
